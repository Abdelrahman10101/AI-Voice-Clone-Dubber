import os
import sys
import json
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rich.console import Console
from rich.panel import Panel
from huggingface_hub import hf_hub_download
from src.config import config, AUDAR_MODELS_DIR, OPENVOICE_MODELS_DIR

console = Console()

def download_openvoice():
    console.print("\n[bold cyan]1/3 Downloading OpenVoice v2 Checkpoints (~110 MB)...[/]")
    converter_dir = OPENVOICE_MODELS_DIR / "checkpoints_v2" / "converter"
    converter_dir.mkdir(parents=True, exist_ok=True)
    
    files = ["checkpoint.pth", "config.json"]
    for filename in files:
        target_file = converter_dir / filename
        if target_file.exists():
            console.print(f"  ✓ {filename} already exists on disk.")
            continue
        
        console.print(f"  Downloading {filename} (with tqdm progress)...")
        hf_hub_download(
            repo_id="myshell-ai/OpenVoiceV2",
            filename=f"converter/{filename}",
            local_dir=str(OPENVOICE_MODELS_DIR / "checkpoints_v2")
        )
        console.print(f"  ✓ {filename} downloaded.")

def download_audar_asr():
    console.print("\n[bold cyan]2/3 Downloading Audar-ASR-V1-Turbo GGUF & Audio Projector (~1.8 GB)...[/]")
    AUDAR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    files = ["Audar-ASR-V1-Turbo-Q4_K_M.gguf", "mmproj-Audar-ASR-V1-Turbo.gguf"]
    for filename in files:
        target_file = AUDAR_MODELS_DIR / filename
        if target_file.exists():
            console.print(f"  ✓ {filename} already exists on disk.")
            continue

        try:
            console.print(f"  Downloading {filename} (with tqdm progress)...")
            hf_hub_download(
                repo_id="audarai/Audar-ASR-V1-Turbo",
                filename=filename,
                local_dir=str(AUDAR_MODELS_DIR)
            )
            console.print(f"  ✓ {filename} downloaded successfully.")
        except Exception as e:
            console.print(f"  [yellow]Notice on {filename}:[/] {e}")
            console.print("  `faster-whisper` (large-v3-turbo) will be used automatically as fallback.")

def pull_ollama_with_tqdm(model_name: str = config.translation_model):
    console.print(f"\n[bold cyan]3/3 Pulling Ollama Model ({model_name}) with tqdm progress...[/]")
    url = f"{config.ollama_url}/api/pull"
    
    # Check if already installed
    try:
        tags_res = requests.get(f"{config.ollama_url}/api/tags", timeout=3)
        if tags_res.status_code == 200:
            installed = [m["name"] for m in tags_res.json().get("models", [])]
            if any(model_name in name for name in installed):
                console.print(f"  ✓ Model '{model_name}' is already installed in Ollama.")
                return
    except Exception:
        console.print("  [bold red]Error:[/] Could not connect to Ollama on http://localhost:11434. Make sure Ollama app is running.")
        return

    # Stream download with tqdm
    try:
        response = requests.post(url, json={"name": model_name}, stream=True, timeout=300)
        response.raise_for_status()

        pbar = None
        current_digest = None

        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            status = data.get("status", "")
            total = data.get("total", 0)
            completed = data.get("completed", 0)
            digest = data.get("digest", "")

            if total > 0:
                if pbar is None or digest != current_digest:
                    if pbar is not None:
                        pbar.close()
                    current_digest = digest
                    pbar = tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"  {status[:25]}",
                        leave=True
                    )
                pbar.n = completed
                pbar.refresh()
            else:
                if status:
                    console.print(f"  status: {status}")

        if pbar is not None:
            pbar.close()
        console.print(f"  ✓ Ollama model '{model_name}' pulled successfully.")

    except Exception as e:
        console.print(f"  [yellow]Failed to stream download via Ollama API ({e}).[/]")
        console.print(f"  You can pull it directly in terminal via: [bold green]ollama pull {model_name}[/]")

def main():
    console.print(Panel(
        "[bold green]STS Studio - All-in-One Model Downloader[/]\n"
        "Downloads all required AI models with live tqdm progress bars into your local storage.",
        border_style="green"
    ))
    
    download_openvoice()
    download_audar_asr()
    pull_ollama_with_tqdm()

    console.print("\n[bold green]🎉 All models are ready! You can now run the pipeline.[/]\n")

if __name__ == "__main__":
    main()
