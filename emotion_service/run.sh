#!/bin/bash

# stop immediately if any command fails.
set -e 

# Extract the directory where this script is located
cd "$(dirname "$0")"

VENV_PATH=".venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

# Pip is smart; if the packages are already installed, it will take 0.5 seconds to verify and move on.
echo "Syncing dependencies..."
pip install -r requirements.txt

echo "Starting Emotion Service..."
cd Gesture_Recognizer
python3 run_camera.py