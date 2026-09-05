import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from src.config import config, OUTPUT_DIR
from src import db
from src import audio_utils
from src import stt
from src import translator
from src import tts_cloner

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def generate_job_id(input_path: str) -> str:
    stem = Path(input_path).stem.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}"

def run_pipeline(
    input_file: str,
    job_id: str = None,
    output_dir: str = None,
    stt_backend: str = config.stt_backend,
    model_llm: str = config.translation_model,
    resume: bool = False
):
    start_time = time.time()
    input_path = Path(input_file).resolve()
    if not input_path.exists():
        console.print(f"[bold red]Error:[/] Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Initialize SQLite database
    db.init_db()

    # Determine Job ID and Output Directory
    if not job_id:
        if resume:
            # Look for existing job matching this input file
            with db.get_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT id FROM jobs WHERE input_file = ? ORDER BY created_at DESC LIMIT 1", (str(input_path),))
                row = cur.fetchone()
                if row:
                    job_id = row["id"]
                    console.print(f"[bold green]Found existing job for resume:[/] {job_id}")
                else:
                    job_id = generate_job_id(str(input_path))
        else:
            job_id = generate_job_id(str(input_path))

    job_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / job_id
    chunks_dir = job_output_dir / "chunks"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Header Panel
    console.print(Panel(
        f"[bold cyan]Arabic-to-English Speech Dubbing & Voice Cloning Pipeline[/]\n"
        f"[dim]Input:[/] {input_path.name}\n"
        f"[dim]Job ID:[/] {job_id}\n"
        f"[dim]STT Engine:[/] {stt_backend.upper()} (4-bit GGUF)\n"
        f"[dim]Translation:[/] Ollama ({model_llm})\n"
        f"[dim]Voice Cloner:[/] OpenVoice v2 (< 1.0 GB VRAM)\n"
        f"[dim]Output Folder:[/] {job_output_dir}",
        title="🎙️ Antigravity STS Studio",
        border_style="cyan"
    ))

    # Register/Update Job in SQLite
    db.create_job(
        job_id=job_id,
        input_file=str(input_path),
        output_dir=str(job_output_dir),
        model_stt=stt_backend,
        model_llm=model_llm,
        model_tts="openvoice_v2"
    )

    extracted_wav = job_output_dir / "original_audio.wav"
    speaker_ref_wav = job_output_dir / "speaker_reference.wav"

    # =========================================================================
    # Step 0: Audio Extraction & VAD Segmentation
    # =========================================================================
    console.print("\n[bold yellow]Step 0: Extracting audio and segmenting with VAD...[/]")
    if not extracted_wav.exists():
        with console.status("[cyan]Extracting 16kHz audio with FFmpeg..."):
            audio_utils.extract_audio(str(input_path), str(extracted_wav), sample_rate=config.sample_rate_stt)

    total_duration = audio_utils.get_audio_duration(str(extracted_wav))
    console.print(f"✓ Extracted clean audio. Total media duration: [bold green]{total_duration:.2f} seconds[/]")

    if not speaker_ref_wav.exists():
        audio_utils.extract_speaker_reference(str(extracted_wav), str(speaker_ref_wav), target_duration=8.0)
        console.print(f"✓ Isolated 8-second speaker vocal reference.")

    # Retrieve existing chunks or perform fresh VAD segmentation
    chunks = db.get_chunks(job_id)
    if not chunks:
        with console.status("[cyan]Performing Voice Activity Detection (VAD) sentence splitting..."):
            segments = audio_utils.vad_segment_audio(
                str(extracted_wav),
                str(chunks_dir),
                min_duration=config.min_chunk_duration,
                max_duration=config.max_chunk_duration
            )
            db.save_chunks_metadata(job_id, segments)
            db.update_job_status(job_id, "segmented", total_chunks=len(segments))
            chunks = db.get_chunks(job_id)

    console.print(f"✓ Total speech chunks: [bold green]{len(chunks)}[/]")

    # =========================================================================
    # Step 1: Speech-to-Text (Audar-ASR / faster-whisper)
    # =========================================================================
    console.print(f"\n[bold yellow]Step 1: Transcribing Arabic Speech ({stt_backend.upper()})...[/]")
    db.update_job_status(job_id, "transcribing")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_stt = progress.add_task("Transcribing chunks", total=len(chunks))
        
        def stt_callback(idx, total, text):
            snippet = (text[:35] + "...") if len(text) > 35 else text
            progress.update(task_stt, completed=idx + 1, description=f"Chunk {idx+1}/{total}: {snippet}")

        chunks = stt.process_stt_stage(chunks, job_id, backend=stt_backend, progress_callback=stt_callback)

    # Export Arabic transcripts and subtitles
    ar_txt_path = job_output_dir / "transcript_ar.txt"
    ar_srt_path = job_output_dir / "transcript_ar.srt"
    audio_utils.export_txt(chunks, str(ar_txt_path), text_key="arabic_text")
    audio_utils.export_srt(chunks, str(ar_srt_path), text_key="arabic_text")
    console.print(f"✓ Arabic transcript saved: [cyan]{ar_txt_path.name}[/]")
    console.print(f"✓ Arabic subtitles saved: [cyan]{ar_srt_path.name}[/]")

    # =========================================================================
    # Step 2: Translation (Ollama: Ministral-3B / Qwen-2.5)
    # =========================================================================
    console.print(f"\n[bold yellow]Step 2: Translating to English via Ollama ({model_llm})...[/]")
    db.update_job_status(job_id, "translating")

    # Verify Ollama server
    if not translator.check_ollama_status():
        console.print("[bold red]Warning:[/] Ollama is not running on http://localhost:11434.")
        console.print("Please launch Ollama and pull your model: [bold green]ollama pull ministral-3b[/]")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_mt = progress.add_task("Translating chunks", total=len(chunks))

        def mt_callback(idx, total, text):
            snippet = (text[:35] + "...") if len(text) > 35 else text
            progress.update(task_mt, completed=idx + 1, description=f"Chunk {idx+1}/{total}: {snippet}")

        chunks = translator.process_translation_stage(chunks, job_id, model_name=model_llm, progress_callback=mt_callback)

    # Export English translations and subtitles
    en_txt_path = job_output_dir / "translation_en.txt"
    en_srt_path = job_output_dir / "translation_en.srt"
    audio_utils.export_txt(chunks, str(en_txt_path), text_key="english_text")
    audio_utils.export_srt(chunks, str(en_srt_path), text_key="english_text")
    console.print(f"✓ English translation saved: [cyan]{en_txt_path.name}[/]")
    console.print(f"✓ English subtitles saved: [cyan]{en_srt_path.name}[/]")

    # =========================================================================
    # Step 3: Voice Cloning (OpenVoice v2)
    # =========================================================================
    console.print(f"\n[bold yellow]Step 3: Cloning Voice with OpenVoice v2 (< 1.0 GB VRAM)...[/]")
    db.update_job_status(job_id, "synthesizing")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_tts = progress.add_task("Voice cloning chunks", total=len(chunks))

        def tts_callback(idx, total, path):
            progress.update(task_tts, completed=idx + 1, description=f"Synthesized {idx+1}/{total}")

        chunks = tts_cloner.process_tts_stage(chunks, str(speaker_ref_wav), job_id, progress_callback=tts_callback)

    # =========================================================================
    # Step 4: Audio Stitching & Synchronization
    # =========================================================================
    console.print(f"\n[bold yellow]Step 4: Stitching audio to match original video timeline...[/]")
    db.update_job_status(job_id, "stitching")

    dubbed_wav_path = job_output_dir / "dubbed_english.wav"
    with console.status("[cyan]Aligning audio and padding silence gaps..."):
        audio_utils.stitch_audio_chunks(
            chunks, 
            total_duration_sec=total_duration,
            output_path=str(dubbed_wav_path),
            target_sample_rate=config.sample_rate_tts
        )

    db.update_job_status(job_id, "completed")
    elapsed = time.time() - start_time

    # =========================================================================
    # Step 5: Summary Table
    # =========================================================================
    table = Table(title="🎉 Job Complete - Results Summary", border_style="green")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", style="magenta")
    table.add_column("Details", style="green")

    table.add_row("Dubbed English Audio", str(dubbed_wav_path.name), f"{total_duration:.1f}s synchronized WAV")
    table.add_row("Arabic Transcript", str(ar_txt_path.name), "Full text (.txt)")
    table.add_row("Arabic Subtitles", str(ar_srt_path.name), "VLC timestamped (.srt)")
    table.add_row("English Translation", str(en_txt_path.name), "Full text (.txt)")
    table.add_row("English Subtitles", str(en_srt_path.name), "VLC timestamped (.srt)")
    table.add_row("Speaker Reference", str(speaker_ref_wav.name), "8s isolated vocal sample")

    console.print(table)
    console.print(f"[bold green]Success![/] Total elapsed time: [bold yellow]{elapsed:.1f} seconds[/]")
    console.print(f"Output files stored in: [bold cyan]{job_output_dir}[/]\n")

def main():
    parser = argparse.ArgumentParser(description="Arabic-to-English Speech-to-Speech Translation & Voice Cloning Studio")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input Arabic video or audio file")
    parser.add_argument("--output-dir", "-o", type=str, default=None, help="Custom output directory")
    parser.add_argument("--job-id", type=str, default=None, help="Custom job ID")
    parser.add_argument("--stt-backend", type=str, default=config.stt_backend, choices=["audar", "whisper"], help="STT backend engine")
    parser.add_argument("--model-llm", type=str, default=config.translation_model, help="Ollama model for translation")
    parser.add_argument("--resume", action="store_true", help="Resume an existing interrupted job from SQLite")
    args = parser.parse_args()

    run_pipeline(
        input_file=args.input,
        job_id=args.job_id,
        output_dir=args.output_dir,
        stt_backend=args.stt_backend,
        model_llm=args.model_llm,
        resume=args.resume
    )

if __name__ == "__main__":
    main()
