#!/bin/bash
# Exit immediately if any command fails
set -e 

# Define paths based on your flattened directory structure
RAG_DIR="./rag_service"
FRONTEND_DIR="./frontend"
VENV_PATH="$RAG_DIR/.venv"

echo "=== Bootstrapping RAG System ==="

# 1. Safely handle the virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment in $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
fi

# Activate the environment
source "$VENV_PATH/bin/activate"

# 2. Always sync dependencies (Avoids the Stale Dependencies Trap)
echo "Syncing dependencies for RAG service..."
echo "$RAG_DIR/requirements.txt"
python3 -m pip install -r "$RAG_DIR/requirements.txt"

# 3. Start the Backend API in the background
echo "Starting FastAPI Backend..."
cd "$RAG_DIR"
# The API will host at http://localhost:8000
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 4. Start the Streamlit Frontend UI
echo "Starting Streamlit UI..."
cd ../$FRONTEND_DIR
# This will open http://localhost:8501
python3 -m pip install -r requirements.txt # Ensure frontend deps are met
streamlit run app.py

# Cleanup: If you stop Streamlit (Ctrl+C), kill the background API too
trap "kill $BACKEND_PID" EXIT