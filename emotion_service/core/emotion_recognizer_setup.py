import os
import json
import mediapipe as mp
from datetime import datetime
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'google_mediapipe_models', 'face_landmarker.task')

base_options = python.BaseOptions(model_asset_path=model_path)

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# --- TEMPORAL STATE VARIABLES ---
# 1. Smoothing Buffer (maxlen=30)
emotions_buffer = deque(maxlen=30)

# 2. Baseline Calibration
is_calibrated = False
calibration_data = []
baseline_offsets = {}
CALIBRATION_FRAMES = 150

# Logging Rate Limiter
last_log_time_ms = 0
LOG_INTERVAL_MS = 1000  # 1000 milliseconds = 1 second

# Work in progress
def calibrate_scores(raw_scores):
    """Zeroes out the student's resting face over the first 150 frames."""
    global is_calibrated, baseline_offsets
    
    if is_calibrated:
        calibrated = {}
        for key, val in raw_scores.items():
            offset = baseline_offsets.get(key, 0.0)
            calibrated[key] = max(0.0, val - offset)
        return calibrated

    return raw_scores

def get_smoothed_emotion(raw_emotion):
    """Applies a Mode Filter to prevent flickering."""
    global emotions_buffer
    emotions_buffer.append(raw_emotion)
    return max(set(emotions_buffer), key=list(emotions_buffer).count)

def predict_emotion(scores):
    """Evaluates the calibrated blendshapes against refined student thresholds."""
    if not is_calibrated:
        return "Calibrating...", "calibration in progress"

    # 1. JOY
    # mouthSmileLeft > 0.25 & mouthSmileRight > 0.25 & cheekSquint > 0.15
    if scores.get('mouthSmileLeft', 0) > 0.25 and scores.get('mouthSmileRight', 0) > 0.25 and \
       (scores.get('cheekSquintLeft', 0) > 0.15 or scores.get('cheekSquintRight', 0) > 0.15):
        cs = max(scores.get('cheekSquintLeft', 0), scores.get('cheekSquintRight', 0))
        trigger = f"mouthSmile={max(scores.get('mouthSmileLeft', 0), scores.get('mouthSmileRight', 0)):.2f}, cheekSquint={cs:.2f}"
        return "Joy", trigger

    # 2. NOTE-TAKING
    # eyeLookDownLeft > 0.4 & eyeLookDownRight > 0.4 & browInnerUp > 0.15
    if scores.get('eyeLookDownLeft', 0) > 0.4 and scores.get('eyeLookDownRight', 0) > 0.4 and scores.get('browInnerUp', 0) > 0.15:
        trigger = f"eyeLookDown={max(scores.get('eyeLookDownLeft', 0), scores.get('eyeLookDownRight', 0)):.2f}, browInnerUp={scores.get('browInnerUp', 0):.2f}"
        return "Note-Taking", trigger

    # 3. FRUSTRATION
    # browDownLeft > 0.65 & browDownRight > 0.65 & mouthPress > 0.25
    if scores.get('browDownLeft', 0) > 0.65 and scores.get('browDownRight', 0) > 0.65 and \
       (scores.get('mouthPressLeft', 0) > 0.25 or scores.get('mouthPressRight', 0) > 0.25):
        trigger = f"browDown={scores.get('browDownLeft', 0):.2f}"
        return "Frustration", trigger

    # 4. CONFUSION
    # browInnerUp > 0.3 & mouthPucker > 0.2
    if scores.get('browInnerUp', 0) > 0.3 and scores.get('mouthPucker', 0) > 0.2:
        trigger = f"browInnerUp={scores.get('browInnerUp', 0):.2f}, mouthPucker={scores.get('mouthPucker', 0):.2f}"
        return "Confusion", trigger

    # 5. CONCENTRATION
    # eyeSquintLeft > 0.4 & eyeSquintRight > 0.4
    if scores.get('eyeSquintLeft', 0) > 0.4 and scores.get('eyeSquintRight', 0) > 0.4:
        trigger = f"eyeSquint={max(scores.get('eyeSquintLeft', 0), scores.get('eyeSquintRight', 0)):.2f}"
        return "Concentration", trigger

    # 6. BORED
    # eyeLookDownLeft > 0.4 & eyeLookDownRight > 0.4 & jawOpen < 0.1
    if scores.get('eyeLookDownLeft', 0) > 0.4 and scores.get('eyeLookDownRight', 0) > 0.4 and scores.get('jawOpen', 0) < 0.1:
        trigger = f"eyeLookDown={max(scores.get('eyeLookDownLeft', 0), scores.get('eyeLookDownRight', 0)):.2f}, jawOpen={scores.get('jawOpen', 0):.2f}"
        return "Bored", trigger

    return "Neutral", "default"

def print_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    with open('emotion_detections.log', 'a') as f:
        f.write(f"FACIAL RECOGNITION RESULT: {result}\n")

def record_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    global last_log_time_ms, is_calibrated, calibration_data, baseline_offsets
    
    emotion = "Calibrating..."
    progress = 0.0
    raw_scores = {}

    if result.face_blendshapes and result.face_blendshapes[0]:
        raw_scores = {blendshape.category_name: blendshape.score for blendshape in result.face_blendshapes[0]}

        if not is_calibrated:
            calibration_data.append(raw_scores)
            progress = min(1.0, len(calibration_data) / CALIBRATION_FRAMES)
            
            if len(calibration_data) >= CALIBRATION_FRAMES:
                keys = raw_scores.keys()
                for key in keys:
                    baseline_offsets[key] = sum(frame.get(key, 0.0) for frame in calibration_data) / len(calibration_data)
                is_calibrated = True
                with open('emotion_detections.log', 'a') as f:
                    f.write(f"\n--- CALIBRATION COMPLETE at {timestamp_ms}ms ---\n")
            emotion = "Calibrating..."
        else:
            progress = 1.0
            raw_emotion, trigger = predict_emotion(raw_scores)
            emotion = get_smoothed_emotion(raw_emotion)

            # Log the specific triggers for the detected emotion
            log_line = f"Emotion: {raw_emotion} (Trigger: {trigger})\n"
            with open('emotion_detections.log', 'a') as f:
                f.write(log_line)

            # --- THE 1-SECOND LOGGING THROTTLE ---
            if timestamp_ms - last_log_time_ms >= LOG_INTERVAL_MS:
                key_scores_str = ", ".join(f"{k}: {v:.4f}" for k, v in raw_scores.items() if k in [
                    'browDownLeft', 'browDownRight', 'mouthPressLeft', 'mouthPressRight', 
                    'browInnerUp', 'mouthPucker', 'eyeSquintLeft', 'eyeSquintRight'
                ])
                with open('emotion_detections.log', 'a') as f:
                    f.write(f"The subject is {emotion} at {timestamp_ms}ms. Progress: {progress:.2f}. Raw key scores: {key_scores_str}\n")
                
                last_log_time_ms = timestamp_ms

    payload = {
        "student_emotion": emotion,
        "calibration_progress": progress,
        "raw_scores": raw_scores
    }
    
    try:
        with open('current_state.json', 'w') as f:
            json.dump(payload, f)
    except Exception:
        pass

def create_emotion_recognizer(running_mode=VisionRunningMode.LIVE_STREAM):
    if running_mode == VisionRunningMode.LIVE_STREAM:
        options = FaceLandmarkerOptions(
            base_options = base_options,
            running_mode = running_mode,
            output_face_blendshapes = True,
            result_callback=record_result
        )
    else:
        options = FaceLandmarkerOptions(
            base_options = base_options,
            running_mode = running_mode,
            output_face_blendshapes = True
        )
    with open('emotion_detections.log', 'a') as f:
        f.write(f"{datetime.now()}\n")
    return FaceLandmarker.create_from_options(options)