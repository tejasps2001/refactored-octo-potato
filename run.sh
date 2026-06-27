#!/bin/bash
# Exit immediately if any command fails
set -e 

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
EMOTION_DIR="$ROOT_DIR/emotion_service"
RAG_DIR="$ROOT_DIR/rag_service"
FRONTEND_DIR="$ROOT_DIR/frontend"

RUN_EMOTION_API=false
RUN_CAMERA=false
RUN_RAG=false
RUN_FRONTEND=false
TARGETED_MODE=false

show_help() {
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "If no options are provided, the script runs ALL services simultaneously."
    echo ""
    echo "Options:"
    echo "  -e, --emotion-api   Launch the FastAPI Emotion State Backend (Port 8001)"
    echo "  -c, --camera        Launch the MediaPipe Webcam Tracking Loop"
    echo "  -r, --rag           Launch the FastAPI Core RAG Backend Pipeline (Port 8000)"
    echo "  -f, --frontend      Launch the Streamlit User Interface View (Port 8501) on the browser"
    echo "  -h, --help          Show this diagnostic utility menu"
    exit 0
}

# Parse command line inputs
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -e|--emotion-api) RUN_EMOTION_API=true; TARGETED_MODE=true; shift ;;
        -c|--camera)      RUN_CAMERA=true;      TARGETED_MODE=true; shift ;;
        -r|--rag)         RUN_RAG=true;         TARGETED_MODE=true; shift ;;
        -f|--frontend)    RUN_FRONTEND=true;    TARGETED_MODE=true; shift ;;
        -h|--help)        show_help ;;
        *) echo "Wrong parameter passed: $1"; echo "Run ./run_all.sh --help for options."; exit 1 ;;
    esac
done

# If no specific flags were requested, run everything
if [ "$TARGETED_MODE" = false ]; then
    RUN_EMOTION_API=true
    RUN_CAMERA=true
    RUN_RAG=true
    RUN_FRONTEND=true
fi

# python3.14 breaks compatibility with the core ML frameworks. Therefore, check for python versions 3.9-3.12
CHOSEN_PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [ "$VERSION" = "3.12" ] || [ "$VERSION" = "3.11" ] || [ "$VERSION" = "3.10" ] || [ "$VERSION" = "3.9" ]; then
            CHOSEN_PYTHON="$cmd"
            echo "Selected python version: $cmd (Python $VERSION)"
            break
        fi
    fi
done

if [ -z "$CHOSEN_PYTHON" ]; then
    echo "ERROR: Could not locate a compatible host runtime (Requires Python 3.9 - 3.12)."
    exit 1
fi

# Kill ghost processes upon exit
trap 'echo -e "\n Shutting down..."; kill $(jobs -p) 2>/dev/null || true; echo "Worspace is cleared and ports are opened."' EXIT

# Activate environments only for those that are actively needed
if [ "$RUN_EMOTION_API" = true ] || [ "$RUN_CAMERA" = true ]; then
    echo "Inspecting environment boundaries for: emotion_service..."
    cd "$EMOTION_DIR"
    if [ ! -d ".venv" ]; then $CHOSEN_PYTHON -m venv .venv; fi
    source .venv/bin/activate && pip install --upgrade pip >/dev/null 2>&1 || true
    pip install -r requirements.txt && deactivate
fi

if [ "$RUN_RAG" = true ]; then
    echo "Inspecting environment boundaries for: rag_service..."
    cd "$RAG_DIR"
    if [ ! -d ".venv" ]; then $CHOSEN_PYTHON -m venv .venv; fi
    source .venv/bin/activate && pip install --upgrade pip >/dev/null 2>&1 || true
    pip install -r requirements.txt && deactivate
fi

if [ "$RUN_FRONTEND" = true ]; then
    echo "Inspecting environment boundaries for: frontend..."
    cd "$FRONTEND_DIR"
    if [ ! -d ".venv" ]; then $CHOSEN_PYTHON -m venv .venv; fi
    source .venv/bin/activate && pip install --upgrade pip >/dev/null 2>&1 || true
    pip install -r requirements.txt && deactivate
fi

# Start Emotion State Broadcaster API
if [ "$RUN_EMOTION_API" = true ]; then
    cd "$EMOTION_DIR"
    source .venv/bin/activate
    # Run in the background if other services are following; else run in the foreground
    if [ "$RUN_CAMERA" = true ] || [ "$RUN_RAG" = true ] || [ "$RUN_FRONTEND" = true ]; then
        python3 -m uvicorn emotion_api:app --host 0.0.0.0 --port 8001 >/dev/null 2>&1 &
        echo "Background Thread: Emotion Broadcaster Service active on Port 8001"
        deactivate
    else
        echo "Foreground Thread: Launching Emotion Broadcaster (Port 8001)..."
        python3 -m uvicorn emotion_api:app --host 0.0.0.0 --port 8001
    fi
fi

# Start MediaPipe Camera Tracking Loop
if [ "$RUN_CAMERA" = true ]; then
    cd "$EMOTION_DIR"
    source .venv/bin/activate
    if [ "$RUN_RAG" = true ] || [ "$RUN_FRONTEND" = true ]; then
        python3 run_camera.py >/dev/null 2>&1 &
        echo "Background Thread: Camera Vision Loop streaming telemetry"
        deactivate
    else
        echo "Foreground Thread: Launching Camera Loop View window..."
        python3 run_camera.py
    fi
fi

# Start FastAPI RAG Backend
if [ "$RUN_RAG" = true ]; then
    cd "$RAG_DIR"
    source .venv/bin/activate
    
    echo "Running automated ingestion pipeline..."
    python3 ingest.py
    
    if [ "$RUN_FRONTEND" = true ]; then
        python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 >/dev/null 2>&1 &
        echo "Background Thread: Vector Engine API active on Port 8000"
        deactivate
    else
        echo "Foreground Thread: Launching Core RAG Engine API (Port 8000)..."
        python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
    fi
fi

# Start Streamlit User Interface
if [ "$RUN_FRONTEND" = true ]; then
    cd "$FRONTEND_DIR"
    source .venv/bin/activate
    echo "Foreground Thread: Launching Streamlit Interface at http://localhost:8501"
    python3 -m streamlit run app.py --server.port 8501
fi