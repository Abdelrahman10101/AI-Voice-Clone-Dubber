import os
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directories
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
AUDAR_MODELS_DIR = MODELS_DIR / "audar"
OPENVOICE_MODELS_DIR = MODELS_DIR / "openvoice"
DB_PATH = DATA_DIR / "history.db"

# Ensure essential directories exist
for directory in [DATA_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR, AUDAR_MODELS_DIR, OPENVOICE_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware & CUDA configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

@dataclass
class PipelineConfig:
    # Audio Settings
    sample_rate_stt: int = 16000
    sample_rate_tts: int = 24000
    min_chunk_duration: float = 3.0   # seconds
    max_chunk_duration: float = 12.0  # seconds
    vad_silence_duration: float = 0.5 # pause length to split sentences
    
    # STT Settings
    stt_backend: str = "audar"  # 'audar' (Arabic-first generative) or 'whisper'
    audar_repo_id: str = "audarai/Audar-ASR-V1-Turbo"
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8_float16" # Fits inside 4GB VRAM
    
    # Translation Settings (Ollama)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    translation_model: str = os.getenv("TRANSLATION_MODEL", "ministral-3:3b")
    fallback_model: str = "qwen2.5:3b"
    ollama_keep_alive: int = 0  # Crucial: Unloads model from RAM/VRAM instantly
    
    # TTS / Voice Cloning Settings
    tts_engine: str = "openvoice_v2" # OpenVoice v2 Decoupled Pipeline
    openvoice_checkpoint_dir: Path = OPENVOICE_MODELS_DIR

config = PipelineConfig()
