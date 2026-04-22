#!/bin/bash

VENV_PATH="./tf"

# Check if the venv directory already exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    python3 -m pip install -r requirements.txt
else
    echo "Virtual environment already exists. Skipping the installation."
    source "$VENV_PATH/bin/activate"
fi

cd ./app/Gesture_Recognizer
python3 ./run_camera.py
