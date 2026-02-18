import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '/home/tejasps/Documents/Internship/app/google_mediapipe_models/gesture_recognizer.task'

base_options = python.BaseOptions(model_asset_path=model_path)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a gesture recognizer instance with the live stream mode:
def print_result(result:GestureRecognizerResult, output_image:mp.Image, timestamp_ms: int):
    print('Gesture recognition result: {}'.format(result))

def create_gesture_recognizer():
    options = GestureRecognizerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)
    return GestureRecognizer.create_from_options(options)