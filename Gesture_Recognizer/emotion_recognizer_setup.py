import mediapipe as mp
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "../../app/google_mediapipe_models/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    with open('emotion_detections.log', 'a') as f:
        f.write(f"FACIAL RECOGNITION RESULT: {result}\n")


def record_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    with open('emotion_detections.log', 'a') as f:
        if(result.face_blendshapes and result.face_blendshapes[0]):
        # No need to check for the emptiness of the inner list because if a
        # face was detected, then there'll be at least one Category object
        # inside.
            f.write(f"The subject is {predict_emotion(result)}.\n")

def predict_emotion(result):
    # Convert list of categories to a dict for easy access
    scores = {blendshape.category_name: blendshape.score for blendshape in result.face_blendshapes[0]}
    # LOGIC RULES
    if scores['mouthSmileRight'] > 0.6 and scores['mouthSmileLeft'] > 0.6: #
        return "in joy"

    if scores['browDownLeft'] > 0.6 and scores['browDownRight'] > 0.6:
        # Distinguish Concentration vs. Frustration
        # No need to check for mouthPressRight too
        if scores['mouthPressLeft'] > 0.6 or scores['mouthFrownLeft'] > 0.6:
            return "is frustrated"
        else: #
            return "is concentrating"

    if scores['browInnerUp'] > 0.6 and scores['browDownRight'] > 0.6:
        return "is confused"

    if scores['browOuterUpLeft'] > 0.6 and scores['browOuterUpRight'] > 0.6:
        return "is engaged"
        
    if scores['eyeLookDownLeft'] > 0.6 or scores['eyeLookDownRight'] > 0.6:
        return "is bored"

    return "is neutral"

def create_emotion_recognizer():
    options = FaceLandmarkerOptions(
        base_options = base_options,
        running_mode = VisionRunningMode.LIVE_STREAM,
        output_face_blendshapes = True,
        result_callback=record_result
    )
    with open('emotion_detections.log', 'a') as f:
        f.write(f"{datetime.now()}\n")
    return FaceLandmarker.create_from_options(options)
