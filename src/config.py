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

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    env_file = BASE_DIR / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

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
    
    # Gemini API Settings
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_stt_model: str = os.getenv("GEMINI_STT_MODEL", "gemini-3.5-transcribe")
    gemini_api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    
    # STT Settings ('gemini', 'whisper', 'audar')
    stt_backend: str = os.getenv("STT_BACKEND", "gemini")
    audar_repo_id: str = "audarai/Audar-ASR-V1-Turbo"
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8_float16" # Fits inside 4GB VRAM
    
    # Translation Settings ('gemini_api' or 'ollama')
    translation_engine: str = os.getenv("TRANSLATION_ENGINE", "gemini_api")
    translation_model: str = os.getenv("TRANSLATION_MODEL", "gemma-4-31b-it")
    fallback_model: str = "gemma-4-31b-it"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_keep_alive: int = 0  # Crucial: Unloads local model from RAM/VRAM instantly
    
    # Free Tier Limits & Batching Settings
    # STT Limit: 3 req/min & 26 req/day -> Batch 10 chunks per call, 21s interval
    stt_batch_size: int = int(os.getenv("STT_BATCH_SIZE", "10"))
    stt_min_interval_sec: float = float(os.getenv("STT_MIN_INTERVAL_SEC", "21.0"))
    
    # Gemma Limit: 30 req/min -> Batch 15 chunks per call, 2.1s interval
    translation_batch_size: int = int(os.getenv("TRANSLATION_BATCH_SIZE", "15"))
    translation_min_interval_sec: float = float(os.getenv("TRANSLATION_MIN_INTERVAL_SEC", "2.1"))

    # TTS / Voice Cloning Settings
    tts_engine: str = "openvoice_v2" # OpenVoice v2 Decoupled Pipeline
    openvoice_checkpoint_dir: Path = OPENVOICE_MODELS_DIR

config = PipelineConfig()
