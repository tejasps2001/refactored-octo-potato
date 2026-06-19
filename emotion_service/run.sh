#!/bin/bash

# stop immediately if any command fails.
set -e 

# Extract the directory where this script is located
cd "$(dirname "$0")"

VENV_PATH=".venv"

if [ -d "$VENV_PATH" ]; then
    echo "Found existing virtual environment."
else
    echo "No virtual environment found. Scanning the host for a compatible Python version..."
    
    CHOSEN_PYTHON=""

    # Install the most latest compatible Python version possible
    for cmd in python3.12 python3.11 python3.10 python3.9 python3; do
        # check if PATH contains the command and send output and err to /dev/null for clean output
        if command -v "$cmd" >/dev/null 2>&1; then 
            # Get the exact python version number
            VERSION=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            if [ "$VERSION" = "3.12" ] || [ "$VERSION" = "3.11" ] || [ "$VERSION" = "3.10" ] || [ "$VERSION" = "3.9" ]; then
                CHOSEN_PYTHON = "$cmd"
                echo "Found compatible python version: $cmd (Python $VERSION)"
                break
            fi
        fi
    done

    # Raise exception if there is no compatible python version
    if [ -z "$CHOSEN_PYTHON" ]; then
        echo "ERROR: Couldn't find a compatible python version (Requires 3.9, 3.10, 3.11 or 3.12)."
        echo "Your default python3 is too new or unsupported. Please install python 3.12 or 3.11."
        exit 1
    fi

    echo "Creating virtual environment..."
    $CHOSEN_PYTHON -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

echo "Syncing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting Emotion Service..."
python3 run_camera.py
