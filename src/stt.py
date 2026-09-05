import os
import gc
import json
import time
import base64
import logging
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.config import config, BASE_DIR, AUDAR_MODELS_DIR
from src.db import update_chunk_stt

logger = logging.getLogger(__name__)


class RateLimiter:
    """Enforces minimum time interval between API requests to respect RPM limits."""
    def __init__(self, min_interval_sec: float):
        self.min_interval = min_interval_sec
        self.last_call_time = 0.0

    def wait(self):
        if self.last_call_time > 0:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.min_interval:
                sleep_sec = self.min_interval - elapsed
                logger.info(f"Rate limiting: waiting {sleep_sec:.1f}s to respect RPM limits...")
                time.sleep(sleep_sec)
        self.last_call_time = time.time()


# Global STT rate limiter (3 RPM -> 21s between calls)
stt_rate_limiter = RateLimiter(config.stt_min_interval_sec)


GEMINI_STT_SINGLE_PROMPT = (
    "You are an expert speech-to-text transcriber specializing in Arabic. "
    "Transcribe the spoken Arabic speech in this audio clip verbatim into accurate written Arabic script. "
    "Accurately capture spoken regional dialects (Egyptian, Levantine, Gulf, Maghrebi, Sudanese, etc.) as well as Modern Standard Arabic. "
    "Include appropriate punctuation (periods, question marks, commas) based on speech pauses. "
    "Do NOT translate, do NOT include explanations, notes, metadata, or quotation marks. "
    "Output ONLY the raw Arabic transcription text."
)

GEMINI_STT_BATCH_PROMPT = (
    "You are an expert speech-to-text transcriber specializing in Arabic. "
    "Below are multiple numbered audio chunks. "
    "Transcribe each chunk verbatim into accurate Arabic script (preserving colloquial dialect vocabulary and punctuation). "
    "You MUST respond with a JSON array where each object has:\n"
    "[\n"
    "  {\"chunk_index\": <int>, \"arabic_text\": \"<Arabic transcription>\"}\n"
    "]\n"
    "Output ONLY the JSON array without any markdown backticks or commentary."
)


class STTEngine:
    """
    Speech-to-Text inference engine supporting:
    1. Gemini Audio Transcriber (API-based, zero local VRAM, batching & rate-limiting optimized).
    2. faster-whisper (CTranslate2) - Universal local fallback.
    3. Audar-ASR-V1-Turbo (GGUF via llama-mtmd-cli) - Arabic-first generative local model.
    """

    def __init__(self, backend: str = config.stt_backend, api_key: Optional[str] = None):
        self.backend = backend
        self.api_key = api_key or config.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.gemini_client = None
        self.model = None
        self.bin_exe = BASE_DIR / "bin" / "llama-mtmd-cli.exe"
        self.audar_model_path = AUDAR_MODELS_DIR / "Audar-ASR-V1-Turbo-Q4_K_M.gguf"
        self.audar_mmproj_path = AUDAR_MODELS_DIR / "mmproj-Audar-ASR-V1-Turbo.gguf"

    def load_model(self):
        """Prepares the selected engine."""
        if self.backend == "gemini":
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please provide it via the .env file, "
                    "environment variable, or --gemini-api-key CLI parameter."
                )
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini Transcriber initialized with model: {config.gemini_stt_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai client ({e}). Will use direct REST API.")
            return

        if self.backend == "audar":
            if self.bin_exe.exists() and self.audar_model_path.exists() and self.audar_mmproj_path.exists():
                logger.info("Using Audar-ASR-V1-Turbo GGUF via llama-mtmd-cli.")
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
                logger.warning(f"CUDA initialization failed ({e}), falling back to CPU for faster-whisper...")
                self.model = WhisperModel(
                    config.whisper_model,
                    device="cpu",
                    compute_type="int8"
                )

    def transcribe_chunks_batch(self, batch_chunks: List[Dict[str, Any]]) -> Dict[int, str]:
        """
        Transcribes multiple audio chunks in a single API call to minimize request count.
        Respects free tier limits (3 RPM / 26 RPD).
        """
        if self.backend != "gemini":
            return {c["chunk_index"]: self.transcribe_chunk(c["chunk_audio_path"]) for c in batch_chunks}

        # gemini-3.5-transcribe is a dedicated speech model that does not support JSON mode
        if "transcribe" in config.gemini_stt_model.lower():
            results = {}
            for c in batch_chunks:
                results[c["chunk_index"]] = self._transcribe_single_gemini(c["chunk_audio_path"])
            return results

        if len(batch_chunks) == 1:
            chunk = batch_chunks[0]
            return {chunk["chunk_index"]: self._transcribe_single_gemini(chunk["chunk_audio_path"])}

        stt_rate_limiter.wait()
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Prepare multimodal payload with multiple audio parts
                contents = [GEMINI_STT_BATCH_PROMPT]
                for c in batch_chunks:
                    idx = c["chunk_index"]
                    audio_path = c["chunk_audio_path"]
                    with open(audio_path, "rb") as f:
                        data = f.read()

                    contents.append(f"--- AUDIO CHUNK {idx} ---")
                    if self.gemini_client is not None:
                        from google.genai import types
                        contents.append(types.Part.from_bytes(data=data, mime_type="audio/wav"))
                    else:
                        # Direct REST fallback
                        b64_data = base64.b64encode(data).decode("utf-8")
                        contents.append({"inline_data": {"mime_type": "audio/wav", "data": b64_data}})

                raw_text = ""
                if self.gemini_client is not None:
                    from google.genai import types
                    resp = self.gemini_client.models.generate_content(
                        model=config.gemini_stt_model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.0
                        )
                    )
                    raw_text = resp.text or ""
                else:
                    import requests
                    url = f"{config.gemini_api_base}/models/{config.gemini_stt_model}:generateContent?key={self.api_key}"
                    parts_payload = []
                    for item in contents:
                        if isinstance(item, str):
                            parts_payload.append({"text": item})
                        elif isinstance(item, dict):
                            parts_payload.append(item)

                    payload = {
                        "contents": [{"parts": parts_payload}],
                        "generationConfig": {
                            "responseMimeType": "application/json",
                            "temperature": 0.0
                        }
                    }
                    resp = requests.post(url, json=payload, timeout=120)
                    resp.raise_for_status()
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        raw_text = "".join([p.get("text", "") for p in parts])

                # Parse JSON array response
                cleaned = raw_text.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                parsed_list = json.loads(cleaned)
                results = {}
                for item in parsed_list:
                    if isinstance(item, dict) and "chunk_index" in item and "arabic_text" in item:
                        results[int(item["chunk_index"])] = str(item["arabic_text"]).strip()

                # Verify all chunks in batch were transcribed
                for c in batch_chunks:
                    idx = c["chunk_index"]
                    if idx not in results:
                        logger.warning(f"Chunk {idx} missing in batch response. Falling back to single transcription.")
                        results[idx] = self._transcribe_single_gemini(c["chunk_audio_path"])

                return results

            except Exception as e:
                logger.warning(f"Batch transcription attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                else:
                    logger.error("Batch transcription failed, falling back to sequential single chunk calls.")
                    fallback_results = {}
                    for c in batch_chunks:
                        fallback_results[c["chunk_index"]] = self._transcribe_single_gemini(c["chunk_audio_path"])
                    return fallback_results

        return {}

    def _transcribe_single_gemini(self, audio_path: str) -> str:
        """Transcribes a single audio chunk using Google Gemini API with retries and rate limiting."""
        stt_rate_limiter.wait()
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                is_transcribe_model = "transcribe" in config.gemini_stt_model.lower()
                text = ""

                if self.gemini_client is not None:
                    from google.genai import types
                    contents = [types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")]
                    if not is_transcribe_model:
                        contents.append(GEMINI_STT_SINGLE_PROMPT)

                    response = self.gemini_client.models.generate_content(
                        model=config.gemini_stt_model,
                        contents=contents
                    )

                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            if hasattr(part, "audio_transcription") and part.audio_transcription:
                                text += getattr(part.audio_transcription, "text", "") or ""
                            elif hasattr(part, "text") and part.text:
                                text += part.text or ""
                    if not text:
                        text = response.text or ""
                else:
                    import requests
                    url = f"{config.gemini_api_base}/models/{config.gemini_stt_model}:generateContent?key={self.api_key}"
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    parts_payload = [{"inline_data": {"mime_type": "audio/wav", "data": audio_b64}}]
                    if not is_transcribe_model:
                        parts_payload.append({"text": GEMINI_STT_SINGLE_PROMPT})

                    payload = {"contents": [{"parts": parts_payload}]}
                    resp = requests.post(url, json=payload, timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for p in parts:
                            if "audioTranscription" in p and isinstance(p["audioTranscription"], dict):
                                text += p["audioTranscription"].get("text", "")
                            elif "text" in p:
                                text += p.get("text", "")

                text = text.strip()
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
                return text

            except Exception as e:
                logger.warning(f"Single transcription attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.error(f"Single transcription failed after {max_retries} attempts: {e}")
                    raise

        return ""

    def transcribe_chunk(self, audio_path: str) -> str:
        """Transcribes a single audio chunk into Arabic text."""
        if self.backend == "gemini":
            return self._transcribe_single_gemini(audio_path)

        elif self.backend == "audar":
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
                return " ".join(text_lines).strip()
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
            return " ".join([seg.text.strip() for seg in segments]).strip()

        return ""

    def unload(self):
        """Forces complete unloading from VRAM/RAM for local models."""
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
        self.gemini_client = None
        logger.info("STT Model / client unloaded.")


def process_stt_stage(
    chunks: List[Dict[str, Any]], 
    job_id: str,
    backend: str = config.stt_backend,
    api_key: Optional[str] = None,
    batch_size: int = config.stt_batch_size,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Processes all speech chunks through the selected STT engine.
    Uses batching to maximize throughput and stay well within free tier limits (26 RPD / 3 RPM).
    """
    engine = STTEngine(backend=backend, api_key=api_key)
    engine.load_model()

    # Find pending chunks that need transcription
    pending_chunks = [c for c in chunks if not (c.get("arabic_text") and c["arabic_text"].strip())]
    completed_chunks = {c["chunk_index"]: c for c in chunks if c.get("arabic_text") and c["arabic_text"].strip()}

    if progress_callback:
        for c in completed_chunks.values():
            progress_callback(c["chunk_index"], len(chunks), c["arabic_text"])

    try:
        # Group pending chunks into batches
        for i in range(0, len(pending_chunks), batch_size):
            batch = pending_chunks[i:i + batch_size]
            batch_results = engine.transcribe_chunks_batch(batch)

            for chunk in batch:
                idx = chunk["chunk_index"]
                arabic_text = batch_results.get(idx, "")
                chunk["arabic_text"] = arabic_text
                update_chunk_stt(job_id, idx, arabic_text)
                completed_chunks[idx] = chunk

                if progress_callback:
                    progress_callback(idx, len(chunks), arabic_text)
    finally:
        engine.unload()

    # Return all chunks ordered by index
    return [completed_chunks[c["chunk_index"]] for c in chunks]
