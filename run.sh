#!/bin/bash

# Create a venv and install from the requirements.txt
python3 -m venv ./ai_tutor
source ai_tutor/bin/activate
python3 -m pip install -r requirements.txt

cd /Gesture_Recognizer

python3 run_camera.py
