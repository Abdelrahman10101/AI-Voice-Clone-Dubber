import os
import gc
import logging
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path
from src.config import config, BASE_DIR, AUDAR_MODELS_DIR
from src.db import update_chunk_stt

logger = logging.getLogger(__name__)

class STTEngine:
    """
    Speech-to-Text inference engine supporting:
    1. Audar-ASR-V1-Turbo (GGUF via llama-mtmd-cli) - Arabic-first generative model.
    2. faster-whisper (CTranslate2) - Universal optimized fallback.
    """

    def __init__(self, backend: str = config.stt_backend):
        self.backend = backend
        self.model = None
        self.bin_exe = BASE_DIR / "bin" / "llama-mtmd-cli.exe"
        self.audar_model_path = AUDAR_MODELS_DIR / "Audar-ASR-V1-Turbo-Q4_K_M.gguf"
        self.audar_mmproj_path = AUDAR_MODELS_DIR / "mmproj-Audar-ASR-V1-Turbo.gguf"

    def load_model(self):
        """Prepares the engine."""
        if self.backend == "audar":
            if self.bin_exe.exists() and self.audar_model_path.exists() and self.audar_mmproj_path.exists():
                logger.info(f"Using Audar-ASR-V1-Turbo GGUF via llama-mtmd-cli on GPU/CPU.")
                return
            else:
                logger.warning("Audar binary or models missing. Falling back to faster-whisper.")
                self.backend = "whisper"

        if self.backend == "whisper":
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading faster-whisper ({config.whisper_model})...")
                self.model = WhisperModel(
                    config.whisper_model,
                    device="cuda",
                    compute_type=config.whisper_compute_type
                )
                logger.info("faster-whisper loaded successfully into VRAM.")
            except Exception as e:
                from faster_whisper import WhisperModel
                logger.warning(f"CUDA failed ({e}), using CPU for faster-whisper...")
                self.model = WhisperModel(
                    config.whisper_model,
                    device="cpu",
                    compute_type="int8"
                )

    def transcribe_chunk(self, audio_path: str) -> str:
        """Transcribes a single audio chunk into Arabic text."""
        if self.backend == "audar":
            cmd = [
                str(self.bin_exe),
                "-m", str(self.audar_model_path),
                "--mmproj", str(self.audar_mmproj_path),
                "--audio", str(audio_path),
                "-p", "فرغ الكلام العربي التالي:",
                "--temp", "0",
                "-ngl", "99"
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
                lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
                text_lines = [
                    l for l in lines 
                    if not l.startswith(('0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', 'WARN', '<|', 'You are', 'Hello', 'Hi there', 'How are'))
                ]
                text = " ".join(text_lines).strip()
                return text
            except Exception as e:
                logger.error(f"Error executing Audar-ASR binary: {e}")
                return ""

        elif self.backend == "whisper":
            if not self.model:
                self.load_model()
            segments, info = self.model.transcribe(
                str(audio_path),
                language="ar",
                beam_size=5,
                temperature=0.0,
                initial_prompt="حوار باللغة العربية والعامية المصرية، فتاوى، إيطاليا، مصر، عمل، باجتياز الاختبار، الوجه الأكمل."
            )
            text = " ".join([seg.text.strip() for seg in segments]).strip()
            return text

        return ""

    def unload(self):
        """Forces complete unloading from VRAM/RAM."""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        logger.info("STT Model unloaded.")

def process_stt_stage(
    chunks: List[Dict[str, Any]], 
    job_id: str,
    backend: str = config.stt_backend,
    progress_callback=None
) -> List[Dict[str, Any]]:
    engine = STTEngine(backend=backend)
    engine.load_model()

    processed_chunks = []
    try:
        for chunk in chunks:
            if chunk.get("arabic_text") and chunk["arabic_text"].strip():
                processed_chunks.append(chunk)
                if progress_callback:
                    progress_callback(chunk["chunk_index"], len(chunks), chunk["arabic_text"])
                continue

            audio_path = chunk["chunk_audio_path"]
            arabic_text = engine.transcribe_chunk(audio_path)
            chunk["arabic_text"] = arabic_text
            
            update_chunk_stt(job_id, chunk["chunk_index"], arabic_text)
            processed_chunks.append(chunk)

            if progress_callback:
                progress_callback(chunk["chunk_index"], len(chunks), arabic_text)
    finally:
        engine.unload()

    return processed_chunks
