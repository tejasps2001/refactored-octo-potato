# Emotionally Aware RAG Tutor

A localized, multi-modal Retrieval-Augmented Generation (RAG) pedagogy tutor that integrates real-time computer vision and speech transcription to observe student engagement and dynamically adapt its pedagogical responses.

## Overview

This system combines a local syllabus-bound RAG assistant (powered by Ollama and LangChain) with a real-time face tracking service (MediaPipe). By monitoring the student's facial expressions and video navigation behavior, the tutor identifies learning struggles, logs engagement telemetry to a SQLite database, adjusts its pedagogical explanation style, and prompts the student with concept-check assessments to reinforce understanding.

## Features

* **Syllabus-Bound Local RAG**: Restricts LLM responses strictly to retrieved course lecture transcripts and notes. Adapts explanation tone and detail based on the student's detected emotion (e.g., patient walkthroughs for frustration/confusion, concise technical responses for high concentration/engagement).
* **Automated Ingestion Pipeline**: Automatically processes PDFs, text notes, and video audio tracks (using Whisper transcriptions) on startup, chunking and embedding the content into ChromaDB.
* **Struggle Tracking & Analytics**: Logs navigation actions (rewinds/scrubs), and emotion timelines into SQLite. Analyzes temporal data to detect repeated review patterns and automatically generates concept-check questions addressing high-struggle topics.
* **Unified Streamlit Interface**: Provides a synchronized video player, real-time researcher metrics (latency, current playhead, extracted emotion state), interactive chat, and a post-session assessment dashboard.
* **Stabilized Telemetry & Simulation Mode**: Uses non-blocking polling endpoints. Includes an automatic offline fallback mode that activates when the camera process is stopped or telemetry data becomes stale, defaulting the UI gracefully to neutral simulation metrics.
* **Unified Orchestrator**: Simplifies startup via a root controller script (`run.sh`) handling environment verification, virtual env builds, port cleanup traps, and multi-service lifecycle control.

## Tech Stack

| Category | Technology |
| --- | --- |
| Inference & LLM | Ollama (Gemma 3 4B, Nomic Embed Text) |
| Ingestion & Transcription | Faster-Whisper (CPU INT8), PyPDF, ChromaDB |
| RAG Framework | LangChain Expression Language (LCEL) |
| APIs & Storage | FastAPI, SQLite3 (Thread-Isolated Async Writes) |
| Vision & Tracking | MediaPipe Task Vision, OpenCV |
| Interface | Streamlit (Non-blocking polling updates) |

## Getting Started

### Prerequisites
* Linux (Ubuntu/Debian recommended)
* Python 3.9 - 3.12
* Ollama with `gemma3:4b` and `nomic-embed-text` installed and running

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/tejasps2001/refactored-octo-potato.git
   cd refactored-octo-potato
   ```

2. Run the unified launcher script:
   Run without flags to start all services:
   ```bash
   ./run.sh
   ```
   Or use the `-e`, `-r`, `-f`, and `-c` flags to start the Emotion API, RAG API, Streamlit Frontend, and Camera Vision Loop respectively:
   ```bash
   ./run.sh -e -r -f -c
   ```

3. Access the Streamlit dashboard:
   Open your browser and navigate to `http://localhost:8501`.

### Unified Launcher Flags
* `-e`: Spawns the Emotion Broadcaster API (Port 8001)
* `-r`: Spawns the RAG Vector Engine API (Port 8000)
* `-f`: Spawns the Streamlit Interface (Port 8501)
* `-c`: Spawns the MediaPipe Camera Loop
* `-h`: Shows the help message

## License

This project is licensed under the MIT License.
