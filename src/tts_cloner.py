import os
import gc
import logging
import asyncio
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.config import config, OPENVOICE_MODELS_DIR
from src.db import update_chunk_tts

logger = logging.getLogger(__name__)

class AudioCloner:
    """
    High-Quality Voice Synthesizer & Cloner:
    1. Generates fluent English speech using Neural Edge-TTS.
    2. Uses OpenVoice v2 Tone Color Converter to morph the vocal timbre
       to match the uploaded speaker reference audio!
    """

    def __init__(self, checkpoint_dir: Path = OPENVOICE_MODELS_DIR):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = "cpu" # ToneColorConverter is tiny (~130MB) and fast on CPU
        self.tone_color_converter = None
        self.source_se = None
        self.target_se = None
        self.default_voice = "en-US-GuyNeural"

    def load_model(self, reference_audio_path: str):
        """Loads OpenVoice Tone Color Converter and extracts speaker vocal embedding."""
        converter_config = self.checkpoint_dir / "checkpoints_v2" / "converter" / "config.json"
        converter_ckpt = self.checkpoint_dir / "checkpoints_v2" / "converter" / "checkpoint.pth"

        if converter_ckpt.exists() and os.path.exists(reference_audio_path):
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                from openvoice.api import ToneColorConverter
                from openvoice import se_extractor

                self.tone_color_converter = ToneColorConverter(str(converter_config), device=self.device)
                self.tone_color_converter.load_ckpt(str(converter_ckpt))
                logger.info("OpenVoice Tone Color Converter loaded successfully.")

                # Extract Target Speaker Embedding from uploaded video audio
                cache_dir = self.checkpoint_dir / "se_cache"
                os.makedirs(cache_dir, exist_ok=True)
                self.target_se, _ = se_extractor.get_se(
                    reference_audio_path,
                    self.tone_color_converter,
                    target_dir=str(cache_dir)
                )
                logger.info("Target speaker vocal timbre extracted successfully.")

                # Pre-compute Source Base Embedding (GuyNeural) once
                base_sample_path = cache_dir / "guy_base_sample.wav"
                if not base_sample_path.exists():
                    import edge_tts
                    sample_text = "Hello and welcome everyone. Today we are presenting a complete technical demonstration of speech translation and voice synthesis step by step."
                    asyncio.run(edge_tts.Communicate(sample_text, self.default_voice).save(str(base_sample_path)))

                self.source_se, _ = se_extractor.get_se(
                    str(base_sample_path),
                    self.tone_color_converter,
                    target_dir=str(cache_dir)
                )
                logger.info("Source speaker base timbre calibrated.")

            except Exception as e:
                logger.warning(f"Voice cloning initialization note ({e}). Falling back to neural voice.")

    def synthesize_and_clone(self, english_text: str, output_path: str) -> str:
        """
        Synthesizes English speech and applies voice cloning.
        """
        if not english_text or not english_text.strip():
            silence = np.zeros(int(config.sample_rate_tts * 0.5), dtype=np.float32)
            sf.write(output_path, silence, config.sample_rate_tts)
            return output_path

        temp_base_wav = str(Path(output_path).with_suffix(".base.wav"))

        # 1. Synthesize base English speech using Edge-TTS
        synthesized = False
        try:
            import edge_tts
            communicate = edge_tts.Communicate(english_text, self.default_voice)
            asyncio.run(communicate.save(temp_base_wav))
            synthesized = True
        except Exception as e:
            logger.warning(f"Edge-TTS synthesis error: {e}")

        if not synthesized or not os.path.exists(temp_base_wav):
            raise RuntimeError("TTS generation failed. Please ensure edge-tts is installed.")

        # 2. Apply Tone Color Converter to morph voice into the uploaded speaker
        if self.tone_color_converter is not None and self.target_se is not None and self.source_se is not None:
            try:
                self.tone_color_converter.convert(
                    audio_src_path=temp_base_wav,
                    src_se=self.source_se,
                    tgt_se=self.target_se,
                    output_path=output_path,
                    tau=0.3
                )
                if os.path.exists(temp_base_wav):
                    os.remove(temp_base_wav)
                return output_path
            except Exception as e:
                logger.warning(f"Tone conversion note ({e}). Keeping base neural voice.")

        # Fallback to clear neural voice if converter wasn't active
        if os.path.exists(temp_base_wav):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.replace(temp_base_wav, output_path)

        return output_path

    def unload(self):
        """Clears memory."""
        del self.tone_color_converter
        self.tone_color_converter = None
        gc.collect()

def process_tts_stage(
    chunks: List[Dict[str, Any]], 
    reference_audio_path: str,
    job_id: str,
    output_dir: Optional[str] = None,
    progress_callback=None
) -> List[Dict[str, Any]]:
    cloner = AudioCloner()
    cloner.load_model(reference_audio_path)

    processed_chunks = []
    try:
        for chunk in chunks:
            # Check if valid cloned audio already exists (resume logic)
            if chunk.get("cloned_audio_path") and os.path.exists(chunk["cloned_audio_path"]):
                if os.path.getsize(chunk["cloned_audio_path"]) > 5000:
                    processed_chunks.append(chunk)
                    if progress_callback:
                        progress_callback(chunk["chunk_index"], len(chunks), chunk["cloned_audio_path"])
                    continue

            english_text = chunk.get("english_text", "").strip()
            chunk_idx = chunk["chunk_index"]
            chunk_dir = output_dir if output_dir else os.path.dirname(chunk.get("chunk_audio_path", "."))
            cloned_audio_path = os.path.join(chunk_dir, f"cloned_{chunk_idx:04d}.wav")

            cloner.synthesize_and_clone(english_text, cloned_audio_path)

            chunk["cloned_audio_path"] = cloned_audio_path
            update_chunk_tts(job_id, chunk_idx, cloned_audio_path)
            processed_chunks.append(chunk)

            if progress_callback:
                progress_callback(chunk_idx, len(chunks), cloned_audio_path)
    finally:
        cloner.unload()

    return processed_chunks
