import mediapipe as mp
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '../../app/google_mediapipe_models/gesture_recognizer.task'

base_options = python.BaseOptions(model_asset_path=model_path)

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result:GestureRecognizerResult, output_image:mp.Image, timestamp_ms: int):
    print('Gesture recognition result: {}'.format(result))

def record_result(result:GestureRecognizerResult, output_image:mp.Image, timestamp_ms: int):
    # Result contains all the hand recognition metadata
    with open('gesture_detections.log', 'a') as f:
        if(result.handedness and result.handedness[0]):
            # No need to check for the emptiness of the inner list because if a
            # hand was detected, then there'll be at least one Category object
            # inside.
            if(result.handedness[0][0].display_name == 'Right'):
                f.write(f"Right hand raised at {timestamp_ms/1000} seconds.\n")
            if(result.handedness[0][0].display_name == 'Left'):
                f.write(f"Left hand raised at {timestamp_ms/1000} seconds.\n")


# Create a gesture recognizer instance with the live stream mode:
def create_gesture_recognizer():
    options = GestureRecognizerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=record_result)
    # Create a new heading for separation
    with open('gesture_detections.log', 'a') as f:
        f.write(f"\n{datetime.now()}\n")
    return GestureRecognizer.create_from_options(options)