import os
import time
import requests
import json
import logging
from typing import List, Dict, Any, Optional
from src.config import config
from src.db import update_chunk_translation

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
                logger.info(f"Translation rate limiting: waiting {sleep_sec:.1f}s to respect 30 RPM limit...")
                time.sleep(sleep_sec)
        self.last_call_time = time.time()


# Global Translation rate limiter (30 RPM -> 2.1s between calls)
translation_rate_limiter = RateLimiter(config.translation_min_interval_sec)


TRANSLATION_SYSTEM_PROMPT = """You are an expert Arabic-to-English translator specializing in spoken regional dialects and audiovisual dubbing.
Your task is to translate spoken Arabic speech transcripts into natural, fluent, spoken English.
Rules:
1. Preserve the original emotion, intent, colloquial nuance (Egyptian, Levantine, Gulf, Maghrebi, etc.), humor, and casual cadence.
2. The English must sound completely natural when spoken aloud by an English voice actor.
3. Do NOT provide explanations, notes, metadata, or markdown.
4. Output ONLY the raw English translation text."""

BATCH_TRANSLATION_SYSTEM_PROMPT = """You are an expert Arabic-to-English translator specializing in spoken regional dialects and audiovisual dubbing.
You will receive a JSON array of Arabic speech chunks.
Translate each chunk into natural, fluent, spoken English (preserving emotion, colloquial nuance, humor, and dialogue cadence).
You MUST respond with a JSON array in the exact same format:
[
  {"chunk_index": <int>, "english_text": "<English translation>"}
]
Do NOT provide explanations or markdown. Output ONLY the raw JSON array."""


def translate_batch_with_gemini_api(
    batch_chunks: List[Dict[str, Any]],
    model_name: str = config.translation_model,
    api_key: Optional[str] = None
) -> Dict[int, str]:
    """
    Translates a batch of Arabic speech chunks in a single API call to Gemma 4 31B.
    Minimizes request count and respects the 30 RPM limit.
    """
    key = api_key or config.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in .env or pass --gemini-api-key.")

    # Filter out empty chunks
    valid_chunks = [c for c in batch_chunks if c.get("arabic_text", "").strip()]
    if not valid_chunks:
        return {c["chunk_index"]: "" for c in batch_chunks}

    if len(valid_chunks) == 1:
        c = valid_chunks[0]
        return {c["chunk_index"]: translate_with_gemini_api(c["arabic_text"], model_name=model_name, api_key=key)}

    translation_rate_limiter.wait()

    payload_data = [{"chunk_index": c["chunk_index"], "arabic_text": c["arabic_text"]} for c in valid_chunks]
    prompt = f"Translate the following Arabic speech chunks into natural spoken English for dubbing:\n{json.dumps(payload_data, ensure_ascii=False, indent=2)}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            raw_text = ""
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=BATCH_TRANSLATION_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        temperature=0.3,
                        top_p=0.9
                    )
                )
                raw_text = response.text or ""
            except Exception as sdk_err:
                logger.debug(f"google-genai SDK call error ({sdk_err}), using direct REST API...")
                url = f"{config.gemini_api_base}/models/{model_name}:generateContent?key={key}"
                payload = {
                    "system_instruction": {"parts": [{"text": BATCH_TRANSLATION_SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.3,
                        "topP": 0.9
                    }
                }
                resp = requests.post(url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    raw_text = "".join([p.get("text", "") for p in parts])

            # Clean and parse JSON
            cleaned = raw_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            results = {}
            for item in parsed:
                if isinstance(item, dict) and "chunk_index" in item and "english_text" in item:
                    results[int(item["chunk_index"])] = str(item["english_text"]).strip()

            # Ensure all chunks in batch are satisfied
            for c in batch_chunks:
                idx = c["chunk_index"]
                if idx not in results:
                    logger.warning(f"Chunk {idx} missing in batch translation response. Falling back to single call.")
                    results[idx] = translate_with_gemini_api(c.get("arabic_text", ""), model_name=model_name, api_key=key)

            return results

        except Exception as e:
            logger.warning(f"Batch translation attempt {attempt + 1}/{max_retries} with {model_name} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logger.error("Batch translation failed, falling back to single chunk calls.")
                fallback_results = {}
                for c in batch_chunks:
                    fallback_results[c["chunk_index"]] = translate_with_gemini_api(c.get("arabic_text", ""), model_name=model_name, api_key=key)
                return fallback_results

    return {c["chunk_index"]: "" for c in batch_chunks}


def translate_with_gemini_api(
    arabic_text: str,
    model_name: str = config.translation_model,
    api_key: Optional[str] = None
) -> str:
    """Translates a single Arabic text chunk using Gemma 4 31B via the Gemini API."""
    if not arabic_text or not arabic_text.strip():
        return ""

    key = api_key or config.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please set it in .env or pass --gemini-api-key.")

    translation_rate_limiter.wait()
    user_prompt = f"Arabic text: {arabic_text}\nEnglish translation:"
    max_retries = 3

    for attempt in range(max_retries):
        try:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=TRANSLATION_SYSTEM_PROMPT,
                        temperature=0.3,
                        top_p=0.9
                    )
                )
                translated = response.text or ""
            except Exception as sdk_err:
                logger.debug(f"google-genai SDK call error ({sdk_err}), using direct REST API...")
                url = f"{config.gemini_api_base}/models/{model_name}:generateContent?key={key}"
                payload = {
                    "system_instruction": {"parts": [{"text": TRANSLATION_SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "topP": 0.9
                    }
                }
                resp = requests.post(url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    translated = "".join([p.get("text", "") for p in parts])
                else:
                    translated = ""

            translated = translated.strip()
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1].strip()
            if translated.lower().startswith("english:"):
                translated = translated[8:].strip()
            if translated.lower().startswith("english translation:"):
                translated = translated[20:].strip()

            return translated

        except Exception as e:
            logger.warning(f"Translation attempt {attempt + 1}/{max_retries} with {model_name} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                logger.error(f"Failed to translate chunk via {model_name} after {max_retries} attempts: {e}")
                raise


def check_ollama_status(ollama_url: str = config.ollama_url) -> bool:
    """Checks if the local Ollama service is reachable."""
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


def unload_ollama_model(model_name: str = config.translation_model, ollama_url: str = config.ollama_url):
    """Explicitly ejects the Ollama model from RAM/VRAM."""
    try:
        requests.post(f"{ollama_url}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=10)
        logger.info(f"Ollama model '{model_name}' successfully unloaded from memory.")
    except Exception:
        pass


def translate_with_ollama(
    arabic_text: str, 
    model_name: str = config.translation_model,
    ollama_url: str = config.ollama_url,
    keep_alive: str = "5m"
) -> str:
    """Translates Arabic text to spoken English using local Ollama."""
    if not arabic_text or not arabic_text.strip():
        return ""

    prompt = f"{TRANSLATION_SYSTEM_PROMPT}\n\nArabic: {arabic_text}\nEnglish:"

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9
        },
        "keep_alive": keep_alive
    }

    try:
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        translated = result.get("response", "").strip()
        
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1].strip()
        if translated.lower().startswith("english:"):
            translated = translated[8:].strip()
            
        return translated
    except Exception as e:
        logger.error(f"Error calling Ollama translation API: {e}")
        raise RuntimeError(
            f"Failed to translate chunk via Ollama model '{model_name}'. "
            f"Make sure Ollama is running and run 'ollama pull {model_name}'."
        )


def process_translation_stage(
    chunks: List[Dict[str, Any]], 
    job_id: str,
    model_name: str = config.translation_model,
    engine: str = config.translation_engine,
    api_key: Optional[str] = None,
    batch_size: int = config.translation_batch_size,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Translates all pending chunks and updates SQLite.
    Batches chunks (default: 15 chunks/call) to respect the 30 RPM limit and minimize latency.
    """
    pending_chunks = [c for c in chunks if not (c.get("english_text") and c["english_text"].strip())]
    completed_chunks = {c["chunk_index"]: c for c in chunks if c.get("english_text") and c["english_text"].strip()}

    if progress_callback:
        for c in completed_chunks.values():
            progress_callback(c["chunk_index"], len(chunks), c["english_text"])

    try:
        if engine == "gemini_api":
            for i in range(0, len(pending_chunks), batch_size):
                batch = pending_chunks[i:i + batch_size]
                batch_results = translate_batch_with_gemini_api(batch, model_name=model_name, api_key=api_key)

                for chunk in batch:
                    idx = chunk["chunk_index"]
                    en_text = batch_results.get(idx, "")
                    chunk["english_text"] = en_text
                    update_chunk_translation(job_id, idx, en_text)
                    completed_chunks[idx] = chunk

                    if progress_callback:
                        progress_callback(idx, len(chunks), en_text)
        else:
            # Local Ollama translation
            for chunk in pending_chunks:
                arabic_text = chunk.get("arabic_text", "").strip()
                english_text = translate_with_ollama(arabic_text, model_name=model_name) if arabic_text else ""
                chunk["english_text"] = english_text
                update_chunk_translation(job_id, chunk["chunk_index"], english_text)
                completed_chunks[chunk["chunk_index"]] = chunk

                if progress_callback:
                    progress_callback(chunk["chunk_index"], len(chunks), english_text)
    finally:
        if engine == "ollama":
            unload_ollama_model(model_name)

    return [completed_chunks[c["chunk_index"]] for c in chunks]


def translate_arabic_to_english(
    arabic_text: str,
    model_name: str = config.translation_model,
    engine: str = config.translation_engine,
    api_key: Optional[str] = None
) -> str:
    """Convenience helper to translate a single Arabic text chunk."""
    if engine == "gemini_api":
        return translate_with_gemini_api(arabic_text, model_name=model_name, api_key=api_key)
    else:
        return translate_with_ollama(arabic_text, model_name=model_name)
