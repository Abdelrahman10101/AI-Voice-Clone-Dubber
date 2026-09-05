import os
import subprocess
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def extract_audio(input_media_path: str, output_wav_path: str, sample_rate: int = 16000) -> str:
    """Extracts 16kHz mono audio from video/audio using ffmpeg with loudness normalization."""
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_media_path),
        "-vn",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,highpass=f=60", # Broadcast normalization + low rumble filter
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_wav_path)
    ]
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        encoding="utf-8", 
        errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
    return output_wav_path

def get_audio_duration(audio_path: str) -> float:
    """Returns audio file duration in seconds."""
    info = sf.info(str(audio_path))
    return float(info.duration)

def vad_segment_audio(
    audio_path: str, 
    output_chunks_dir: str, 
    min_duration: float = 3.0, 
    max_duration: float = 12.0,
    silence_thresh: Optional[int] = None,
    min_silence_len: int = 400
) -> List[Dict[str, Any]]:
    """
    Splits long audio into sentence-like speech chunks using adaptive silence detection.
    Guarantees chunks stay between min_duration and max_duration to prevent model OOM
    while ensuring no quiet speech is dropped.
    """
    os.makedirs(output_chunks_dir, exist_ok=True)
    sound = AudioSegment.from_file(str(audio_path))
    total_len_ms = len(sound)

    # Dynamic adaptive threshold based on audio loudness
    if silence_thresh is None:
        # 14 dB below average loudness, capped between -32 and -48 dBFS
        computed_thresh = int(sound.dBFS - 14)
        silence_thresh = max(-48, min(-30, computed_thresh))

    # Detect non-silent ranges in milliseconds
    nonsilent_ranges = detect_nonsilent(sound, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if not nonsilent_ranges:
        nonsilent_ranges = [[0, total_len_ms]]

    # Expand ranges with 250ms padding to prevent cutting off words at boundaries
    pad_ms = 250
    padded_ranges = []
    for s_ms, e_ms in nonsilent_ranges:
        s_padded = max(0, s_ms - pad_ms)
        e_padded = min(total_len_ms, e_ms + pad_ms)
        padded_ranges.append((s_padded, e_padded))

    # Merge overlapping or close ranges
    merged_ranges = []
    curr_start, curr_end = padded_ranges[0]

    for start_ms, end_ms in padded_ranges[1:]:
        dur_sec = (end_ms - curr_start) / 1000.0
        # If chunks overlap or adding this keeps it under max_duration, merge
        if start_ms <= curr_end or dur_sec <= max_duration:
            curr_end = max(curr_end, end_ms)
        else:
            merged_ranges.append((curr_start, curr_end))
            curr_start, curr_end = start_ms, end_ms
    merged_ranges.append((curr_start, curr_end))

    # Ensure no large gap is completely dropped (if a gap > 3s exists, add it as a chunk)
    contiguous_ranges = []
    prev_end = 0
    for s_ms, e_ms in merged_ranges:
        gap = s_ms - prev_end
        if gap > 3000:
            # Add the gap as its own chunk in case quiet speech was in between
            contiguous_ranges.append((prev_end, s_ms))
        contiguous_ranges.append((s_ms, e_ms))
        prev_end = e_ms

    if total_len_ms - prev_end > 3000:
        contiguous_ranges.append((prev_end, total_len_ms))

    # Second pass: Split any remaining chunks that exceed max_duration mechanically
    final_ranges = []
    max_ms = int(max_duration * 1000)
    for start_ms, end_ms in contiguous_ranges:
        chunk_len = end_ms - start_ms
        if chunk_len > max_ms:
            for sub_start in range(start_ms, end_ms, max_ms):
                sub_end = min(sub_start + max_ms, end_ms)
                if (sub_end - sub_start) >= 1000: # Only keep chunks >= 1 sec
                    final_ranges.append((sub_start, sub_end))
        else:
            if chunk_len >= 1000:
                final_ranges.append((start_ms, end_ms))

    if not final_ranges:
        final_ranges = [[0, total_len_ms]]

    # Export chunks and assemble metadata
    chunks_meta = []
    for idx, (start_ms, end_ms) in enumerate(final_ranges):
        chunk_audio = sound[start_ms:end_ms]
        chunk_filename = f"chunk_{idx:04d}.wav"
        chunk_path = os.path.join(output_chunks_dir, chunk_filename)
        chunk_audio.export(chunk_path, format="wav")

        chunks_meta.append({
            "index": idx,
            "start": round(start_ms / 1000.0, 3),
            "end": round(end_ms / 1000.0, 3),
            "duration": round((end_ms - start_ms) / 1000.0, 3),
            "audio_path": chunk_path
        })

    return chunks_meta

def extract_speaker_reference(
    audio_path: str, 
    output_ref_path: str, 
    target_duration: float = 12.0
) -> str:
    """
    Extracts a clean 10-12 second vocal segment with high RMS/clarity
    to use as speaker reference for voice cloning.
    """
    sound = AudioSegment.from_file(str(audio_path))
    total_sec = len(sound) / 1000.0
    
    if total_sec <= target_duration:
        sound.export(output_ref_path, format="wav")
        return output_ref_path

    # Search for an 8-second window with the strongest vocal energy
    window_ms = int(target_duration * 1000)
    step_ms = 1000
    best_rms = -1
    best_start = 0

    for start_ms in range(0, len(sound) - window_ms, step_ms):
        segment = sound[start_ms:start_ms + window_ms]
        if segment.rms > best_rms:
            best_rms = segment.rms
            best_start = start_ms

    best_sample = sound[best_start:best_start + window_ms]
    best_sample.export(output_ref_path, format="wav")
    return output_ref_path

def stitch_audio_chunks(
    chunks: List[Dict[str, Any]], 
    total_duration_sec: float, 
    output_path: str,
    target_sample_rate: int = 24000
) -> str:
    """
    Concatenates synthesized English audio chunks matching the original
    video/audio timeline and preserving silence gaps.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_ms = int(total_duration_sec * 1000)
    
    # Initialize silent base audio canvas
    canvas = AudioSegment.silent(duration=max(total_ms, 1000), frame_rate=target_sample_rate)

    for chunk in chunks:
        cloned_path = chunk.get("cloned_audio_path")
        if not cloned_path or not os.path.exists(cloned_path):
            continue

        chunk_audio = AudioSegment.from_file(cloned_path)
        start_ms = int(chunk["start_time"] * 1000)
        
        # Overlay chunk onto canvas at exact start timestamp
        canvas = canvas.overlay(chunk_audio, position=start_ms)

    canvas.export(output_path, format="wav")
    return output_path

def export_srt(chunks: List[Dict[str, Any]], output_path: str, text_key: str = "arabic_text") -> str:
    """Generates a standard .srt subtitle file from chunks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get(text_key, "").strip()
            if not text:
                continue
            start_str = format_timestamp_srt(chunk["start_time"])
            end_str = format_timestamp_srt(chunk["end_time"])
            f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
    return output_path

def export_txt(chunks: List[Dict[str, Any]], output_path: str, text_key: str = "arabic_text") -> str:
    """Generates a plain readable text file from chunks."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            text = chunk.get(text_key, "").strip()
            if text:
                f.write(f"{text}\n")
    return output_path

def mux_video_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """Combines original video with newly dubbed audio into a synchronized MP4."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path)
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg video-audio muxing failed: {result.stderr}")
    return output_path

