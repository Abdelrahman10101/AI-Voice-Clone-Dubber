import requests
import json
import logging
from typing import List, Dict, Any, Optional
from src.config import config
from src.db import update_chunk_translation

logger = logging.getLogger(__name__)

TRANSLATION_SYSTEM_PROMPT = """You are an expert Arabic-to-English translator specializing in spoken dialects and audiovisual dubbing.
Your task is to translate spoken Arabic speech transcript into natural, fluent English.
Rules:
1. Preserve the original emotion, intent, colloquial expressions, and casual cadence of the speaker.
2. The translation should sound natural when read aloud in English.
3. Do NOT provide explanations, translator notes, or extra punctuation.
4. Output ONLY the raw English translation text."""

def check_ollama_status(ollama_url: str = config.ollama_url) -> bool:
    """Checks if the local Ollama service is reachable."""
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=3)
        return res.status_code == 200
    except Exception:
        return False

def get_installed_ollama_models(ollama_url: str = config.ollama_url) -> List[str]:
    """Returns a list of models currently downloaded in Ollama."""
    try:
        res = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if res.status_code == 200:
            data = res.json()
            return [m["name"].split(":")[0] for m in data.get("models", [])]
    except Exception:
        pass
    return []

def unload_ollama_model(model_name: str = config.translation_model, ollama_url: str = config.ollama_url):
    """Explicitly ejects the Ollama model from RAM/VRAM."""
    try:
        requests.post(f"{ollama_url}/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=10)
        logger.info(f"Ollama model '{model_name}' successfully unloaded from memory.")
    except Exception:
        pass

def translate_arabic_to_english(
    arabic_text: str, 
    model_name: str = config.translation_model,
    ollama_url: str = config.ollama_url,
    keep_alive: str = "5m"
) -> str:
    """
    Translates Arabic text to spoken English using local Ollama.
    Keeps model warm during translation stage, then unloads afterward.
    """
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
        # Generous timeout for model cold-start into RAM
        response = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()
        translated = result.get("response", "").strip()
        
        # Clean up any residual markdown or formatting artifacts
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
    progress_callback=None
) -> List[Dict[str, Any]]:
    """
    Translates all pending chunks and updates SQLite.
    Keeps model warm during processing, then unloads it completely.
    """
    translated_chunks = []
    
    try:
        for chunk in chunks:
            # Check if already translated (resume functionality)
            if chunk.get("english_text") and chunk["english_text"].strip():
                translated_chunks.append(chunk)
                if progress_callback:
                    progress_callback(chunk["chunk_index"], len(chunks), chunk["english_text"])
                continue

            arabic_text = chunk.get("arabic_text", "").strip()
            if not arabic_text:
                english_text = ""
            else:
                english_text = translate_arabic_to_english(arabic_text, model_name=model_name, keep_alive="5m")

            chunk["english_text"] = english_text
            update_chunk_translation(job_id, chunk["chunk_index"], english_text)
            translated_chunks.append(chunk)

            if progress_callback:
                progress_callback(chunk["chunk_index"], len(chunks), english_text)
    finally:
        # Crucial: Unload model from RAM/VRAM immediately after the translation stage finishes!
        unload_ollama_model(model_name)

    return translated_chunks
