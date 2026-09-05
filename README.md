# 🎙️ Arabic-to-English Speech-to-Speech (STS) Voice Cloning Studio

An end-to-end Speech-to-Speech dubbing pipeline that translates spoken Arabic (including regional colloquial dialects) into English while **preserving the original speaker's vocal tone and timbre**.

Optimized for consumer hardware with **12 GB RAM and 4 GB VRAM (NVIDIA GPU)** by combining cloud-accelerated intelligence (**Google Gemini Transcriber** & **Gemma 4 31B**) with lightweight local voice cloning (**OpenVoice v2**).

---

## 🌟 Key Features

* **Dialect-Aware Audio Transcription:** Powered by **Google Gemini 3.5 Transcriber** (`gemini-3.5-flash`) for dialect accuracy across Egyptian, Levantine, Gulf, Maghrebi, and MSA speech without consuming local VRAM. (Local `whisper` and `audar` fallbacks supported).
* **Gemma 4 31B Translation:** Powered by Google's flagship **Gemma 4 31B** (`gemma-4-31b-it`) via the Gemini API. Delivers spoken dialogue translations that capture local idioms, emotion, and conversational cadence without local memory bottleneck.
* **Ultra-Light Voice Cloning:** Uses local `OpenVoice v2` decoupled tone conversion (< 1.0 GB VRAM) to match pitch, timbre, and vocal resonance from the Arabic voice.
* **Long-Form Video Dubbing:** Uses FFmpeg and Voice Activity Detection (VAD) to segment 1-hour+ videos into 3–12s natural sentence chunks and stitches dubbed audio back together seamlessly.
* **Subtitles Included:** Automatically outputs matching `.txt` transcripts and timestamped `.srt` subtitles in both Arabic and English.
* **Auto-Resume on Interruption:** Tracks job and chunk states in a local SQLite database (`data/history.db`). If an hour-long video is paused or interrupted, it resumes from the exact chunk where it left off.

---

## 🏗️ Architecture

```
[Arabic Video / Audio]
         │
         ▼
[FFmpeg 16kHz Extraction & VAD Sentence Chunking (3-12s)]
         │
         ▼
[Step 1: STT with Gemini 3.5 Transcriber] ────────> 0 GB VRAM (Cloud API)
         │ (Arabic Text Chunks)
         ▼
[Step 2: Translation with Gemma 4 31B] ────────────> 0 GB VRAM (Cloud API)
         │ (English Text Chunks)
         ▼
[Step 3: Voice Cloning with OpenVoice v2] ─────────> ~0.8 GB VRAM (Local GPU)
         │ (Cloned English Audio Chunks)
         ▼
[Step 4: Audio Alignment & Timeline Stitching]
         │
         ▼
[Output: dubbed_english.wav + transcript_ar.srt + translation_en.srt]
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **FFmpeg**: Must be installed and on your system `PATH`.
- **Python 3.10+**
- **Google Gemini API Key**: Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 2. Configure Environment
Copy `.env.example` to `.env` and set your API key:
```powershell
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_STT_MODEL=gemini-3.5-flash
TRANSLATION_ENGINE=gemini_api
TRANSLATION_MODEL=gemma-4-31b-it
```

### 3. Download TTS Model Weights
Run the included downloader to fetch the voice-cloning weights:
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
| `--stt-backend` | STT engine (`gemini`, `whisper`, `audar`) | `gemini` |
| `--translation-engine` | Translation engine (`gemini_api`, `ollama`) | `gemini_api` |
| `--model-llm` | Model to use for translation | `gemma-4-31b-it` |
| `--gemini-api-key` | Optional inline API key override | Read from `.env` |
| `--resume` | Resume an existing interrupted job from SQLite | `False` |
| `--output-dir`, `-o`| Custom destination folder | `data/output/<job_id>` |
| `--job-id` | Custom identifier for the job | Auto-generated |

### Examples

**Run with Gemini Transcriber & Gemma 4 31B (Default):**
```powershell
python src/pipeline.py --input data/input/my_video.mp4
```

**Resume an interrupted job:**
```powershell
python src/pipeline.py --input data/input/my_video.mp4 --resume
```

**Run 100% Offline with local Whisper & Ollama:**
```powershell
python src/pipeline.py --input data/input/my_video.mp4 --stt-backend whisper --translation-engine ollama --model-llm gemma3:1b
```

---

## 📁 Output Structure

Every run creates a dedicated folder inside `data/output/<job_id>/`:

```text
data/output/my_video_20260905_073000/
├── original_audio.wav       # Extracted 16kHz source audio
├── speaker_reference.wav    # 8-second isolated vocal reference
├── transcript_ar.txt        # Full Arabic transcript
├── transcript_ar.srt        # Arabic subtitles (for VLC)
├── translation_en.txt       # Full English translation
├── translation_en.srt       # English subtitles (for VLC)
├── chunks/                  # Intermediate chunk audio clips (WAV)
└── dubbed_english.wav       # Final synchronized dubbed audio
```

---

## 💡 Memory & Hardware Safety

* **0 GB VRAM & RAM consumed during STT and Translation:** By offloading speech recognition to Gemini 2.5 Flash and translation to Gemma 4 31B, your local machine runs with zero thermal or memory throttling.
* **100% GPU Headroom for Voice Cloning:** Only step 3 (OpenVoice v2) uses the GPU, peaking at under **1.0 GB VRAM**, well within 4 GB cards.
