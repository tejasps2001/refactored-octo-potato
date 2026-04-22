import mediapipe as mp
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "../google_mediapipe_models/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    with open('emotion_detections.log', 'a') as f:
        f.write(f"FACIAL RECOGNITION RESULT: {result}\n")


def record_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    # frame_count = 
    # if result.face_blendshapes and result.face_blendshapes[0]:
        # No need to check for the emptiness of the inner list in result.face_blendshapes[0] 
        # because if a face was detected, then there'll be at least one Category object inside.
            
    # with open('emotion_detections.log', 'a') as f:
    #     f.write(f"The subject is {predict_emotion(result_list)}.\n")
    with open('emotion_detections.log', 'a') as f:
        if result.face_blendshapes and result.face_blendshapes[0]:
        # No need to check for the emptiness of the inner list in result.face_blendshapes[0] 
        # because if a face was detected, then there'll be at least one Category object inside.
            f.write(f"The subject is {predict_emotion(result)}.\n")

def predict_emotion(result):
    # Convert list of categories to a dict for easy access
    scores = {blendshape.category_name: blendshape.score for blendshape in result.face_blendshapes[0]}
    # LOGIC RULES
    # 1. JOY (Duchenne Smile: Mouth + Cheeks + Eyes)
    if scores['mouthSmileLeft'] > 0.5 and scores['mouthSmileRight'] > 0.5 and scores['cheekPuff'] > 0.2:
        return "Joy"

    # 2. FRUSTRATION (Tension in brow, pressed lips, narrowed eyes)
    if scores['browDownLeft'] > 0.6 and scores['browDownRight'] > 0.6 and scores['mouthPressLeft'] > 0.4:
        return "Frustration"

    # 3. CONCENTRATION (Narrowed eyes, slight brow lowering, mouth closed)
    if scores['eyeSquintLeft'] > 0.4 and scores['eyeSquintRight'] > 0.4 and scores['browDownLeft'] > 0.3:
        return "Concentration"

    # 4. CONFUSION (Asymmetrical brow, squinting, or lip pursing)
    if (scores['browInnerUp'] > 0.3 and scores['browDownLeft'] > 0.3) or scores['mouthPucker'] > 0.4:
        return "Confusion"

    # 5. ENGAGED (Widened eyes, slightly raised brows, leaning forward/up)
    if scores['eyeWideLeft'] > 0.3 and scores['eyeWideRight'] > 0.3 and scores['browOuterUpLeft'] > 0.2:
        return "Engaged"

    # 6. BORED (Droopy lids, neutral mouth, slight jaw drop)
    if scores['eyeLookDownLeft'] > 0.4 and scores['eyeLookDownRight'] > 0.4 and scores['jawOpen'] < 0.1:
        return "Bored"

    # 7. NEUTRAL (Default state)
    return "Neutral"
    
def create_emotion_recognizer(running_mode=VisionRunningMode.LIVE_STREAM):
    if running_mode == VisionRunningMode.LIVE_STREAM:
        options = FaceLandmarkerOptions(
            base_options = base_options,
            running_mode = running_mode,
            output_face_blendshapes = True,
            result_callback=record_result
        )
    else:
        # Here, the running_mode is IMAGE
        options = FaceLandmarkerOptions(
            base_options = base_options,
            running_mode = running_mode,
            output_face_blendshapes = True
        )
    with open('emotion_detections.log', 'a') as f:
        f.write(f"{datetime.now()}\n")
    return FaceLandmarker.create_from_options(options)
