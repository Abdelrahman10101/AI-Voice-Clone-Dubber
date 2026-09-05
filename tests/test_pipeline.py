import os
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
from src import db
from src.audio_utils import extract_audio, vad_segment_audio, extract_speaker_reference, stitch_audio_chunks, export_srt, export_txt
from src.config import config

def test_full_flow():
    print("\n--- Running End-to-End STS Flow Verification ---")
    
    # 1. Setup test audio
    test_dir = BASE_DIR / "data" / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_audio_path = test_dir / "sample_arabic.wav"
    sr = 16000
    # Create 8 seconds of synthetic speech-like signal
    t = np.linspace(0, 8, 8 * sr)
    signal = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.normal(0, 0.05, 8 * sr)
    sf.write(str(test_audio_path), signal, sr)
    print("✓ Created sample audio file:", test_audio_path)

    # 2. Test DB Initialization
    db.init_db()
    job_id = "test_job_123"
    db.create_job(job_id, str(test_audio_path), str(test_dir), "whisper", "ministral-3b", "openvoice_v2")
    job = db.get_job(job_id)
    assert job is not None, "Job record not found in SQLite!"
    print("✓ Database job creation verified.")

    # 3. Test Audio Segmentation
    chunks_dir = test_dir / "chunks"
    segments = vad_segment_audio(str(test_audio_path), str(chunks_dir), min_duration=2.0, max_duration=6.0)
    assert len(segments) > 0, "No segments detected by VAD!"
    db.save_chunks_metadata(job_id, segments)
    print(f"✓ VAD segmentation produced {len(segments)} chunks.")

    # 4. Test Speaker Reference Extraction
    ref_path = test_dir / "speaker_ref.wav"
    extract_speaker_reference(str(test_audio_path), str(ref_path), target_duration=4.0)
    assert ref_path.exists(), "Speaker reference file was not created!"
    print("✓ Speaker vocal reference extracted.")

    # 5. Populate and test subtitle exports
    chunks = db.get_chunks(job_id)
    for c in chunks:
        db.update_chunk_stt(job_id, c["chunk_index"], "السلام عليكم ورحمة الله وبركاته")
        db.update_chunk_translation(job_id, c["chunk_index"], "Peace and mercy of God be upon you.")
        # Create a mock cloned audio chunk for testing stitching
        cloned_chunk_path = chunks_dir / f"cloned_{c['chunk_index']:04d}.wav"
        cloned_signal = 0.2 * np.sin(2 * np.pi * 400 * np.linspace(0, c["duration"], int(24000 * c["duration"])))
        sf.write(str(cloned_chunk_path), cloned_signal, 24000)
        db.update_chunk_tts(job_id, c["chunk_index"], str(cloned_chunk_path))

    updated_chunks = db.get_chunks(job_id)
    assert updated_chunks[0]["arabic_text"] != "", "STT DB update failed!"
    assert updated_chunks[0]["english_text"] != "", "Translation DB update failed!"
    assert updated_chunks[0]["cloned_audio_path"] != "", "TTS DB update failed!"
    print("✓ Database chunk updates verified.")

    # 6. Test SRT and TXT export
    ar_srt = test_dir / "transcript_ar.srt"
    en_srt = test_dir / "translation_en.srt"
    export_srt(updated_chunks, str(ar_srt), text_key="arabic_text")
    export_srt(updated_chunks, str(en_srt), text_key="english_text")
    assert ar_srt.exists() and os.path.getsize(str(ar_srt)) > 0, "Arabic SRT missing!"
    assert en_srt.exists() and os.path.getsize(str(en_srt)) > 0, "English SRT missing!"
    print("✓ SRT subtitle generation verified.")

    # 7. Test Stitching
    stitched_wav = test_dir / "dubbed_english.wav"
    stitch_audio_chunks(updated_chunks, total_duration_sec=8.0, output_path=str(stitched_wav), target_sample_rate=24000)
    assert stitched_wav.exists() and os.path.getsize(str(stitched_wav)) > 0, "Stitched audio missing!"
    print("✓ Audio stitching and alignment verified.")

    # 8. Test Resume Logic
    resumed_chunks = db.get_chunks(job_id)
    completed_chunks = [c for c in resumed_chunks if c["status"] == "synthesized"]
    assert len(completed_chunks) == len(resumed_chunks), "All chunks should be marked synthesized!"
    print("✓ Auto-resume status persistence verified.")

    print("\n[SUCCESS] All pipeline unit and integration components passed!")

if __name__ == "__main__":
    test_full_flow()
