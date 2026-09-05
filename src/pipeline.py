import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import config, DATA_DIR, OUTPUT_DIR
from src import audio_utils, db, stt, translator, tts_cloner

# Force UTF-8 on Windows consoles to prevent cp1252 charmap encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("sts_pipeline")


def generate_job_id(input_file_path: str) -> str:
    """Generates a reproducible, clean job identifier based on input file name and timestamp."""
    stem = Path(input_file_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    clean_stem = "".join(c if c.isalnum() else "_" for c in stem)
    return f"{clean_stem}_{timestamp}"


def run_pipeline(
    input_file: str,
    job_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    stt_backend: str = config.stt_backend,
    translation_engine: str = config.translation_engine,
    model_llm: str = config.translation_model,
    gemini_api_key: Optional[str] = None,
    stt_batch_size: int = config.stt_batch_size,
    translation_batch_size: int = config.translation_batch_size,
    resume: bool = False
):
    """Executes the complete Arabic-to-English Speech-to-Speech dubbing pipeline."""
    start_time = time.time()
    input_path = Path(input_file).resolve()
    
    if not input_path.exists():
        console.print(f"[bold red]Error:[/] Input media file not found: {input_path}")
        sys.exit(1)

    effective_api_key = gemini_api_key or config.gemini_api_key or os.getenv("GEMINI_API_KEY", "")

    # Preflight check for Gemini API key
    if (stt_backend == "gemini" or translation_engine == "gemini_api") and not effective_api_key:
        console.print(Panel(
            "[bold red]Missing Google Gemini API Key![/]\n\n"
            "You selected Gemini Transcriber or Gemma 4 31B via Gemini API, but no API key was found.\n"
            "Please provide your API key by:\n"
            "  1. Creating a [bold green].env[/] file in the project root: [bold yellow]GEMINI_API_KEY=your_key_here[/]\n"
            "  2. Setting an environment variable: [bold yellow]$env:GEMINI_API_KEY='your_key_here'[/]\n"
            "  3. Passing the CLI parameter: [bold yellow]--gemini-api-key your_key_here[/]\n\n"
            "[dim]Get a free API key at: https://aistudio.google.com/app/apikey[/]",
            title="Configuration Required",
            border_style="red"
        ))
        sys.exit(1)

    # Initialize SQLite Database
    db.init_db()

    # Determine job ID
    if not job_id:
        if resume:
            latest_job = db.get_latest_job_for_file(str(input_path))
            if latest_job:
                job_id = latest_job["id"]
                console.print(f"[bold green]Found existing job for resume:[/] {job_id}")
            else:
                job_id = generate_job_id(str(input_path))
        else:
            job_id = generate_job_id(str(input_path))

    job_output_dir = Path(output_dir) if output_dir else OUTPUT_DIR / job_id
    chunks_dir = job_output_dir / "chunks"
    job_output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    stt_label = f"Gemini Transcriber ({config.gemini_stt_model})" if stt_backend == "gemini" else stt_backend.upper()
    trans_label = f"Gemini API ({model_llm})" if translation_engine == "gemini_api" else f"Ollama ({model_llm})"

    # Header Panel
    console.print(Panel(
        f"[bold cyan]Arabic-to-English Speech Dubbing & Voice Cloning Studio[/]\n"
        f"[dim]Input:[/] {input_path.name}\n"
        f"[dim]Job ID:[/] {job_id}\n"
        f"[dim]STT Engine:[/] {stt_label}\n"
        f"[dim]Translation:[/] {trans_label}\n"
        f"[dim]Voice Cloner:[/] OpenVoice v2 (< 1.0 GB VRAM)\n"
        f"[dim]Output Folder:[/] {job_output_dir}",
        title="STS Studio",
        border_style="cyan"
    ))

    # Register/Update Job in SQLite
    db.create_job(
        job_id=job_id,
        input_file=str(input_path),
        output_dir=str(job_output_dir),
        model_stt=stt_backend,
        model_llm=f"{translation_engine}:{model_llm}",
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
        console.print("✓ Isolated 8-second speaker vocal reference.")

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
    # Step 1: Speech-to-Text (Gemini Transcriber / whisper / audar)
    # =========================================================================
    console.print(f"\n[bold yellow]Step 1: Transcribing Arabic Speech ({stt_label})...[/]")
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

        chunks = stt.process_stt_stage(
            chunks, 
            job_id, 
            backend=stt_backend, 
            api_key=effective_api_key, 
            batch_size=stt_batch_size,
            progress_callback=stt_callback
        )

    # Export Arabic transcripts and subtitles
    ar_txt_path = job_output_dir / "transcript_ar.txt"
    ar_srt_path = job_output_dir / "transcript_ar.srt"
    audio_utils.export_txt(chunks, str(ar_txt_path), text_key="arabic_text")
    audio_utils.export_srt(chunks, str(ar_srt_path), text_key="arabic_text")
    console.print(f"✓ Arabic transcript saved: [cyan]{ar_txt_path.name}[/]")
    console.print(f"✓ Arabic subtitles saved: [cyan]{ar_srt_path.name}[/]")

    # =========================================================================
    # Step 2: Translation (Gemma 4 31B via Gemini API / Ollama)
    # =========================================================================
    console.print(f"\n[bold yellow]Step 2: Translating to English ({trans_label})...[/]")
    db.update_job_status(job_id, "translating")

    # Verify Ollama server if local engine is chosen
    if translation_engine == "ollama" and not translator.check_ollama_status():
        console.print("[bold red]Warning:[/] Ollama is not running on http://localhost:11434.")
        console.print(f"Please launch Ollama or pull your model: [bold green]ollama pull {model_llm}[/]")
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

        chunks = translator.process_translation_stage(
            chunks, 
            job_id, 
            model_name=model_llm, 
            engine=translation_engine, 
            api_key=effective_api_key, 
            batch_size=translation_batch_size,
            progress_callback=mt_callback
        )

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
        task_tts = progress.add_task("Cloning audio chunks", total=len(chunks))

        def tts_callback(idx, total, *args):
            progress.update(task_tts, completed=idx + 1, description=f"Dubbing Chunk {idx+1}/{total}")

        chunks = tts_cloner.process_tts_stage(
            chunks=chunks, 
            reference_audio_path=str(speaker_ref_wav), 
            job_id=job_id, 
            output_dir=str(chunks_dir), 
            progress_callback=tts_callback
        )

    # =========================================================================
    # Step 4: Stitching & Final Audio Alignment
    # =========================================================================
    console.print("\n[bold yellow]Step 4: Stitching dubbed audio into complete timeline...[/]")
    db.update_job_status(job_id, "stitching")

    dubbed_wav_path = job_output_dir / "dubbed_english.wav"
    with console.status("[cyan]Aligning audio and padding silence gaps..."):
        audio_utils.stitch_audio_chunks(
            chunks, 
            total_duration_sec=total_duration,
            output_path=str(dubbed_wav_path),
            target_sample_rate=config.sample_rate_tts
        )

    dubbed_video_path = None
    if input_path.suffix.lower() in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
        dubbed_video_path = job_output_dir / f"dubbed_{input_path.stem}.mp4"
        with console.status("[cyan]Muxing video with cloned English audio..."):
            audio_utils.mux_video_audio(str(input_path), str(dubbed_wav_path), str(dubbed_video_path))

    db.update_job_status(job_id, "completed")
    elapsed = time.time() - start_time

    # =========================================================================
    # Step 5: Summary Table
    # =========================================================================
    table = Table(title="Job Complete - Results Summary", border_style="green")
    table.add_column("Artifact", style="cyan")
    table.add_column("Path", style="magenta")
    table.add_column("Details", style="green")

    if dubbed_video_path and dubbed_video_path.exists():
        table.add_row("Dubbed English Video", str(dubbed_video_path.name), "Synchronized MP4 video with dubbed audio")
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
    parser.add_argument("--stt-backend", type=str, default=config.stt_backend, choices=["gemini", "audar", "whisper"], help="STT backend engine (default: gemini)")
    parser.add_argument("--translation-engine", type=str, default=config.translation_engine, choices=["gemini_api", "ollama"], help="Translation engine ('gemini_api' or 'ollama')")
    parser.add_argument("--model-llm", type=str, default=config.translation_model, help="Translation model (default: gemma-4-31b-it for gemini_api)")
    parser.add_argument("--gemini-api-key", type=str, default=None, help="Google Gemini API key")
    parser.add_argument("--stt-batch-size", type=int, default=config.stt_batch_size, help=f"Chunks per STT request (default: {config.stt_batch_size})")
    parser.add_argument("--translation-batch-size", type=int, default=config.translation_batch_size, help=f"Chunks per translation request (default: {config.translation_batch_size})")
    parser.add_argument("--resume", action="store_true", help="Resume an existing interrupted job from SQLite")
    args = parser.parse_args()

    run_pipeline(
        input_file=args.input,
        job_id=args.job_id,
        output_dir=args.output_dir,
        stt_backend=args.stt_backend,
        translation_engine=args.translation_engine,
        model_llm=args.model_llm,
        gemini_api_key=args.gemini_api_key,
        stt_batch_size=args.stt_batch_size,
        translation_batch_size=args.translation_batch_size,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
