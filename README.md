# 🎙️ Arabic-to-English Speech-to-Speech (STS) Voice Cloning Studio

An offline, local Speech-to-Speech dubbing pipeline that translates spoken Arabic (including regional dialects) into English while **preserving the original speaker's vocal tone and timbre**. 

Specially optimized for consumer hardware with **12 GB RAM and 4 GB VRAM (NVIDIA GPU)** without external API keys.

---

## 🌟 Key Features

* **Arabic Dialect Comprehension:** Uses `Audar-ASR-V1-Turbo` (GGUF 4-bit) and `faster-whisper` (large-v3-turbo) to accurately transcribe Egyptian, Gulf, Levantine, Maghrebi, and MSA speech.
* **Dialect-Aware Translation:** Powered by `Ministral 3B` (or `Qwen 2.5 3B`) through local Ollama with `"keep_alive": 0` for immediate memory deallocation.
* **Ultra-Light Voice Cloning:** Uses `OpenVoice v2` decoupled tone conversion (< 1.0 GB VRAM) to match pitch, timbre, and vocal resonance from the Arabic voice.
* **Long-Form Video Dubbing:** Uses FFmpeg and Voice Activity Detection (VAD) to segment 1-hour+ videos into natural sentence chunks and stitches dubbed audio back together seamlessly.
* **Subtitles Included:** Generates both `.txt` transcripts and timestamped `.srt` subtitles for VLC.
* **Auto-Resume on Interruption:** Tracks job and chunk states in a local SQLite database (`data/history.db`). If an hour-long video is paused or interrupted, it resumes from the exact chunk where it left off.

---

## 🏗️ Architecture

```
[Arabic Video / Audio]
         │
         ▼
[FFmpeg 16kHz Extraction & VAD Sentence Chunking]
         │
         ▼
[Step 1: STT with Audar-ASR / Whisper Large] ───> ~1.6 GB VRAM ───> Memory Cleared!
         │ (Arabic Text Chunks)
         ▼
[Step 2: Translation with Ollama Ministral 3B] ──> ~2.2 GB RAM  ───> Memory Cleared!
         │ (English Text Chunks)
         ▼
[Step 3: Voice Cloning with OpenVoice v2] ───────> ~0.8 GB VRAM ───> Memory Cleared!
         │ (Cloned English Audio Chunks)
         ▼
[Step 4: Audio Alignment & Stitching]
         │
         ▼
[Output: dubbed_english.wav + translation_en.srt]
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **FFmpeg**: Must be installed and on your system `PATH`.
- **Ollama**: Download and install from [ollama.ai](https://ollama.ai).

### 2. Pull the Translation Model
```powershell
ollama pull ministral-3:3b
```
*(Alternative: `ollama pull qwen2.5:3b`)*

### 3. Download Model Weights
Run the included downloader to fetch the speech and voice-cloning weights:
```powershell
python download_models.py
```

### 4. Run the Dubbing Pipeline
Place your audio or video file inside `data/input/` and run:
```powershell
python src/pipeline.py --input data/input/my_video.mp4
```

---

## 🛠️ CLI Options

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--input`, `-i` | Path to Arabic video or audio file (MP4, MKV, WAV, MP3) | **Required** |
| `--resume` | Resume an existing interrupted job from SQLite | `False` |
| `--model-llm` | Ollama model to use for translation | `ministral-3b` |
| `--stt-backend` | STT engine (`audar` or `whisper`) | `audar` |
| `--output-dir`, `-o`| Custom destination folder | `data/output/<job_id>` |
| `--job-id` | Custom identifier for the job | Auto-generated |

### Examples

**Resume an interrupted job:**
```powershell
python src/pipeline.py --input data/input/my_video.mp4 --resume
```

**Use Qwen 2.5 for translation instead:**
```powershell
python src/pipeline.py --input data/input/my_video.mp4 --model-llm qwen2.5:3b
```

---

## 📁 Output Structure

Every run creates a dedicated folder inside `data/output/<job_id>/`:

```text
data/output/my_video_20260903_001200/
├── original_audio.wav       # Extracted 16kHz source audio
├── speaker_reference.wav    # 8-second isolated vocal reference
├── transcript_ar.txt        # Full Arabic transcript
├── transcript_ar.srt        # Arabic subtitles (for VLC)
├── translation_en.txt       # Full English translation
├── translation_en.srt       # English subtitles (for VLC)
├── chunks/                  # Intermediate chunk audio clips
└── dubbed_english.wav       # Final synchronized dubbed audio
```

---

## 💡 Memory & Hardware Safety

* Peak VRAM never exceeds **1.8 GB** at any point.
* Windows Desktop (DWM) consumes ~0.4 GB, leaving **over 1.8 GB of free VRAM headroom** on a 4GB card.
* Model objects are explicitly deleted, Python garbage collector is triggered, and `torch.cuda.empty_cache()` is called between stages.
