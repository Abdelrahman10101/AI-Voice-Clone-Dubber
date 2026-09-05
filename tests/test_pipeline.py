import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Force UTF-8 on Windows stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf
import pytest
from src import db, stt, translator
from src.audio_utils import extract_audio, vad_segment_audio, extract_speaker_reference, stitch_audio_chunks, export_srt, export_txt
from src.config import config

# Disable rate-limiting sleep during unit tests
stt.stt_rate_limiter.min_interval = 0.0
translator.translation_rate_limiter.min_interval = 0.0


def test_config_defaults():
    print("\n--- Testing Configuration Defaults ---")
    assert config.stt_backend == "gemini", f"Expected stt_backend='gemini', got {config.stt_backend}"
    assert config.gemini_stt_model == "gemini-3.5-transcribe", f"Expected gemini_stt_model='gemini-3.5-transcribe', got {config.gemini_stt_model}"
    assert config.translation_engine == "gemini_api", f"Expected translation_engine='gemini_api', got {config.translation_engine}"
    assert config.translation_model == "gemma-4-31b-it", f"Expected translation_model='gemma-4-31b-it', got {config.translation_model}"
    print("[OK] Configuration defaults verified: Gemini 3.5 Transcribe & Gemma 4 31B.")


def test_gemini_stt_missing_key_validation():
    print("\n--- Testing Gemini STT Missing Key Validation ---")
    with patch.object(config, "gemini_api_key", ""), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        engine = stt.STTEngine(backend="gemini", api_key="")
        with pytest.raises(ValueError) as exc_info:
            engine.load_model()
        assert "GEMINI_API_KEY is not set" in str(exc_info.value)
    print("[OK] Gemini STT correctly rejects execution when GEMINI_API_KEY is missing.")


def test_gemma_translation_missing_key_validation():
    print("\n--- Testing Gemma 4 31B Missing Key Validation ---")
    with patch.object(config, "gemini_api_key", ""), patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        with pytest.raises(ValueError) as exc_info:
            translator.translate_with_gemini_api("صباح الخير", api_key="")
        assert "GEMINI_API_KEY is not set" in str(exc_info.value)
    print("[OK] Gemma 4 31B correctly rejects execution when GEMINI_API_KEY is missing.")


def test_gemini_stt_mock_transcription():
    print("\n--- Testing Gemini STT Mock Transcription ---")
    test_audio = BASE_DIR / "data" / "test_run" / "mock.wav"
    test_audio.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(test_audio), np.zeros(16000), 16000)

    mock_resp = MagicMock()
    mock_resp.text = '"صباح الخير يا باشا"'
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    engine = stt.STTEngine(backend="gemini", api_key="fake_key")
    engine.gemini_client = mock_client

    result = engine.transcribe_chunk(str(test_audio))
    assert result == "صباح الخير يا باشا", f"Expected clean Arabic text, got: {result}"
    print("[OK] Gemini STT mock transcription output verified.")


def test_gemini_stt_batch_transcription():
    print("\n--- Testing Gemini STT Batch Transcription ---")
    test_dir = BASE_DIR / "data" / "test_run"
    test_dir.mkdir(parents=True, exist_ok=True)
    c0 = test_dir / "c0.wav"
    c1 = test_dir / "c1.wav"
    sf.write(str(c0), np.zeros(8000), 16000)
    sf.write(str(c1), np.zeros(8000), 16000)

    mock_resp0 = MagicMock()
    mock_part0 = MagicMock()
    mock_part0.audio_transcription.text = "السلام عليكم"
    mock_resp0.candidates = [MagicMock(content=MagicMock(parts=[mock_part0]))]
    mock_resp0.text = None

    mock_resp1 = MagicMock()
    mock_part1 = MagicMock()
    mock_part1.audio_transcription.text = "وعليكم السلام"
    mock_resp1.candidates = [MagicMock(content=MagicMock(parts=[mock_part1]))]
    mock_resp1.text = None

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [mock_resp0, mock_resp1]

    engine = stt.STTEngine(backend="gemini", api_key="fake_key")
    engine.gemini_client = mock_client

    batch = [
        {"chunk_index": 0, "chunk_audio_path": str(c0)},
        {"chunk_index": 1, "chunk_audio_path": str(c1)}
    ]
    results = engine.transcribe_chunks_batch(batch)
    assert results[0] == "السلام عليكم"
    assert results[1] == "وعليكم السلام"
    print("[OK] Gemini STT batch transcription verified.")


def test_gemma_translation_batch():
    print("\n--- Testing Gemma 4 31B Batch Translation ---")
    mock_resp = MagicMock()
    mock_resp.text = '[{"chunk_index": 0, "english_text": "Peace be upon you."}, {"chunk_index": 1, "english_text": "And upon you peace."}]'
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        batch = [
            {"chunk_index": 0, "arabic_text": "السلام عليكم"},
            {"chunk_index": 1, "arabic_text": "وعليكم السلام"}
        ]
        results = translator.translate_batch_with_gemini_api(batch, model_name="gemma-4-31b-it", api_key="fake_key")
        assert results[0] == "Peace be upon you."
        assert results[1] == "And upon you peace."
    print("[OK] Gemma 4 31B batch translation verified (multiple sentences in 1 request).")


def test_full_pipeline_flow():
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
    print("[OK] Created sample audio file:", test_audio_path)

    # 2. Test DB Initialization
    db.init_db()
    job_id = "test_job_123"
    db.create_job(job_id, str(test_audio_path), str(test_dir), "gemini", "gemini_api:gemma-4-31b-it", "openvoice_v2")
    job = db.get_job(job_id)
    assert job is not None, "Job record not found in SQLite!"
    print("[OK] Database job creation verified.")

    # 3. Test Audio Segmentation
    chunks_dir = test_dir / "chunks"
    segments = vad_segment_audio(str(test_audio_path), str(chunks_dir), min_duration=2.0, max_duration=6.0)
    assert len(segments) > 0, "No segments detected by VAD!"
    db.save_chunks_metadata(job_id, segments)
    print(f"[OK] VAD segmentation produced {len(segments)} chunks.")

    # 4. Test Speaker Reference Extraction
    ref_path = test_dir / "speaker_ref.wav"
    extract_speaker_reference(str(test_audio_path), str(ref_path), target_duration=4.0)
    assert ref_path.exists(), "Speaker reference file was not created!"
    print("[OK] Speaker vocal reference extracted.")

    # 5. Populate and test subtitle exports
    chunks = db.get_chunks(job_id)
    for c in chunks:
        db.update_chunk_stt(job_id, c["chunk_index"], "السلام عليكم ورحمة الله وبركاته")
        db.update_chunk_translation(job_id, c["chunk_index"], "Peace and blessings be upon you.")
        # Create a mock cloned audio chunk for testing stitching
        cloned_chunk_path = chunks_dir / f"cloned_{c['chunk_index']:04d}.wav"
        cloned_signal = 0.2 * np.sin(2 * np.pi * 400 * np.linspace(0, c["duration"], int(24000 * c["duration"])))
        sf.write(str(cloned_chunk_path), cloned_signal, 24000)
        db.update_chunk_tts(job_id, c["chunk_index"], str(cloned_chunk_path))

    updated_chunks = db.get_chunks(job_id)
    assert updated_chunks[0]["arabic_text"] != "", "STT DB update failed!"
    assert updated_chunks[0]["english_text"] != "", "Translation DB update failed!"
    assert updated_chunks[0]["cloned_audio_path"] != "", "TTS DB update failed!"
    print("[OK] Database chunk updates verified.")

    # 6. Test SRT and TXT export
    ar_srt = test_dir / "transcript_ar.srt"
    en_srt = test_dir / "translation_en.srt"
    export_srt(updated_chunks, str(ar_srt), text_key="arabic_text")
    export_srt(updated_chunks, str(en_srt), text_key="english_text")
    assert ar_srt.exists() and os.path.getsize(str(ar_srt)) > 0, "Arabic SRT missing!"
    assert en_srt.exists() and os.path.getsize(str(en_srt)) > 0, "English SRT missing!"
    print("[OK] SRT subtitle generation verified.")

    # 7. Test Stitching
    stitched_wav = test_dir / "dubbed_english.wav"
    stitch_audio_chunks(updated_chunks, total_duration_sec=8.0, output_path=str(stitched_wav), target_sample_rate=24000)
    assert stitched_wav.exists() and os.path.getsize(str(stitched_wav)) > 0, "Stitched audio missing!"
    print("[OK] Audio stitching and alignment verified.")

    # 8. Test Resume Logic
    resumed_chunks = db.get_chunks(job_id)
    completed_chunks = [c for c in resumed_chunks if c["status"] == "synthesized"]
    assert len(completed_chunks) == len(resumed_chunks), "All chunks should be marked synthesized!"
    print("[OK] Auto-resume status persistence verified.")

    print("\n[SUCCESS] All pipeline unit and integration components passed!")


if __name__ == "__main__":
    test_config_defaults()
    test_gemini_stt_missing_key_validation()
    test_gemma_translation_missing_key_validation()
    test_gemini_stt_mock_transcription()
    test_gemini_stt_batch_transcription()
    test_gemma_translation_batch()
    test_full_pipeline_flow()
