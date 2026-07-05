# Emotionally Aware RAG Tutor

A localized, multi-modal Retrieval-Augmented Generation (RAG) pedagogy tutor that integrates real-time computer vision and speech transcription to observe student engagement and dynamically adapt its pedagogical responses.

## Overview

This system combines a local syllabus-bound RAG assistant (powered by Ollama and LangChain) with a real-time face tracking service (MediaPipe). By monitoring the student's facial expressions and video navigation behavior, the tutor identifies learning struggles, logs engagement telemetry to a SQLite database, adjusts its pedagogical explanation style, and prompts the student with concept-check assessments to reinforce understanding.

## Features

- **Syllabus-Bound Local RAG**: Restricts LLM responses strictly to retrieved course lecture transcripts and notes. Adapts explanation tone and detail based on the student's detected emotion (e.g., patient walkthroughs for frustration/confusion, concise technical responses for high concentration/engagement).
- **Automated Ingestion Pipeline**: Automatically processes PDFs, text notes, and video audio tracks (using Whisper transcriptions) on startup, chunking and embedding the content into ChromaDB.
- **Struggle Tracking & Analytics**: Logs navigation actions (rewinds/scrubs), and emotion timelines into SQLite. Analyzes temporal data to detect repeated review patterns and automatically generates concept-check questions addressing high-struggle topics.
- **Unified Streamlit Interface**: Provides a synchronized video player, real-time researcher metrics (latency, current playhead, extracted emotion state), interactive chat, and a post-session assessment dashboard.
- **Stabilized Telemetry & Simulation Mode**: Uses non-blocking polling endpoints. Includes an automatic offline fallback mode that activates when the camera process is stopped or telemetry data becomes stale, defaulting the UI gracefully to neutral simulation metrics.
- **Unified Orchestrator**: Simplifies startup via a cross-platform root controller script (run_all.py) handling environment verification, virtual env builds, port cleanup, and multi-service lifecycle control.

## Tech Stack

| Category | Technology |
| --- | --- |
| Inference & LLM | Ollama (Gemma 3 4B, Nomic Embed Text) |
| Ingestion & Transcription | Faster-Whisper (CPU INT8), PyPDF, ChromaDB |
| RAG Framework | LangChain Expression Language (LCEL) |
| APIs & Storage | FastAPI, SQLite3 (Thread-Isolated Async Writes) |
| Vision & Tracking | MediaPipe Task Vision, OpenCV |
| Interface | Streamlit (Non-blocking polling updates) |

## Setup & Running the System

Follow these steps in chronological order to initialize and run the emotionally aware RAG tutor.

### Prerequisites (Manual Setup)

1. **Ollama Environment**:
   - Download and install Ollama for your operating system from the official website: https://ollama.com.
   - Run the Ollama application to ensure the background service is active (an icon will appear in your system tray).

2. **Learning Materials**:
   - Place your raw learning materials (e.g., lecture PDFs, course notes, or video files like `lecture.mp4`) inside the `rag_service/data/` directory.
   - Note: The directory must contain at least one document or media file before initiating the orchestration script.

### Execution & Automated Orchestration

1. **Start the Unified Launcher**:
   Open a terminal in the root directory of the project and run the orchestrator script:
   - On Windows:
     ```cmd
     python run_all.py
     ```
   - On Linux/macOS:
     ```bash
     python3 run_all.py
     ```

2. **Automated Setup Steps**:
   Upon launch, the orchestrator script dynamically executes the following automated routines:
   - **Environment Synchronization**: Programmatically creates virtual environments (`.venv`) for each microservice (`emotion_service`, `rag_service`, and `frontend`) and runs `pip install -r requirements.txt` to sync dependencies automatically.
   - **Ollama PATH & Weight Check**: Verifies that Ollama is installed on your system. It then checks if the required model weights (`gemma3:4b` and `nomic-embed-text`) exist locally. If any model is missing, it automatically pulls the weights from the Ollama registry.
   - **Automated Ingestion**: Extracts audio from any video files, runs local transcription using faster-whisper, splits documents into chunks, embeds the text, and stores them in a local ChromaDB database.

3. **Access the Streamlit Dashboard**:
   Once all services show as active in your terminal, open your web browser and navigate to:
   - http://localhost:8501

## Unified Launcher Flags

To run individual services or target specific setup steps manually, run `python run_all.py` with one or more of the following flags:

- `-e`, `--emotion-api`: Spawns the Emotion Broadcaster API (Port 8001)
- `-r`, `--rag`: Spawns the RAG Vector Engine API (Port 8000)
- `-f`, `--frontend`, `--ui`: Spawns the Streamlit Interface (Port 8501)
- `-c`, `--camera`: Spawns the MediaPipe Camera Loop
- `--backend`: Spawns both RAG and Emotion backends
- `-h`, `--help`: Shows the help message

## License

This project is licensed under the MIT License.
