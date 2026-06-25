import json
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

# --- TEMPORAL STATE VARIABLES ---
# 1. Smoothing Buffer
emotions_buffer = []
BUFFER_SIZE = 30  # Approx 1 second of frames at 30fps

# 2. Baseline Calibration
is_calibrated = False
start_time_ms = None
calibration_data = []
baseline_offsets = {}
CALIBRATION_DURATION_MS = 10000  # 10 seconds

# Logging Rate Limiter
last_log_time_ms = 0
LOG_INTERVAL_MS = 1000  # 1000 milliseconds = 1 second

def calibrate_scores(raw_scores, timestamp_ms):
    """Zeroes out the student's resting face over the first 10 seconds."""
    global is_calibrated, baseline_offsets, calibration_data, start_time_ms
    
    # If already calibrated, subtract the baselines from the live scores
    if is_calibrated:
        calibrated = {}
        for key, val in raw_scores.items():
            offset = baseline_offsets.get(key, 0.0)
            calibrated[key] = max(0.0, val - offset)
        return calibrated

    # Initialize start time on the very first frame
    if start_time_ms is None:
        start_time_ms = timestamp_ms

    # Collect data during the calibration window
    calibration_data.append(raw_scores)

    # Check if 10 seconds have passed
    if (timestamp_ms - start_time_ms) >= CALIBRATION_DURATION_MS:
        keys = raw_scores.keys()
        for key in keys:
            if calibration_data:
                baseline_offsets[key] = sum(frame[key] for frame in calibration_data) / len(calibration_data)
            else:
                baseline_offsets[key] = 0.0
        
        is_calibrated = True
        with open('emotion_detections.log', 'a') as f:
            f.write(f"\n--- CALIBRATION COMPLETE at {timestamp_ms}ms ---\n")

    return raw_scores  # Return raw scores while still calibrating

def get_smoothed_emotion(raw_emotion):
    """Applies a Mode Filter to prevent flickering."""
    global emotions_buffer
    
    emotions_buffer.append(raw_emotion)
    
    if len(emotions_buffer) > BUFFER_SIZE:
        emotions_buffer.pop(0)
    
    return max(set(emotions_buffer), key=emotions_buffer.count)

def predict_emotion(scores):
    """Evaluates the calibrated blendshapes against refined student thresholds."""
    
    # 1. THE "NOTE-TAKER" PROBLEM
    # If eyes are down, check if the inner brow is slightly elevated (focusing)
    if scores.get('eyeLookDownLeft', 0) > 0.4 and scores.get('eyeLookDownRight', 0) > 0.4:
        if scores.get('browInnerUp', 0) > 0.15:
            return "Note-Taking"
        elif scores.get('jawOpen', 0) < 0.1:
            return "Bored"

    # 2. ACTIVE FRUSTRATION (Refined for Live Movement)
    # Brow Down + Lip Press
    if (scores.get('browDownLeft', 0) > 0.5 or scores.get('browDownRight', 0) > 0.5) and \
       (scores.get('mouthPressLeft', 0) > 0.15 or scores.get('mouthPressRight', 0) > 0.15):
        return "Frustration"

    # 3. PASSIVE CONFUSION
    # One Brow Up + Mouth Pucker
    if scores.get('browInnerUp', 0) > 0.3 and scores.get('mouthPucker', 0) > 0.2:
        return "Confusion"

    # 4. COGNITIVE LOAD / CONCENTRATION
    # Sustained Eye Squint (Buffer handles the temporal 'sustained' requirement naturally)
    if scores.get('eyeSquintLeft', 0) > 0.4 and scores.get('eyeSquintRight', 0) > 0.4:
        return "Concentration"

    # 5. JOY (Duchenne Smile)
    if scores.get('mouthSmileLeft', 0) > 0.5 and scores.get('mouthSmileRight', 0) > 0.5 and scores.get('cheekPuff', 0) > 0.2:
        return "Joy"

    # 6. ENGAGED
    if scores.get('eyeWideLeft', 0) > 0.3 and scores.get('eyeWideRight', 0) > 0.3 and scores.get('browOuterUpLeft', 0) > 0.2:
        return "Engaged"

    # 7. NEUTRAL
    return "Neutral"

def print_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    with open('emotion_detections.log', 'a') as f:
        f.write(f"FACIAL RECOGNITION RESULT: {result}\n")

def record_result(result:FaceLandmarkerResult, output_image:mp.Image, timestamp_ms:int):
    global last_log_time_ms, start_time_ms
    
    emotion = "Calibrating..."
    progress = 0.0
    raw_scores = {}

    if result.face_blendshapes and result.face_blendshapes[0]:
        raw_scores = {blendshape.category_name: blendshape.score for blendshape in result.face_blendshapes[0]}

        if start_time_ms is not None:
            progress = min(1.0, (timestamp_ms - start_time_ms) / CALIBRATION_DURATION_MS)

        calibrated_scores = calibrate_scores(raw_scores, timestamp_ms)

        if is_calibrated:
            progress = 1.0
            raw_emotion = predict_emotion(calibrated_scores)
            emotion = get_smoothed_emotion(raw_emotion)

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