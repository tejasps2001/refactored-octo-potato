import sys
import os
import cv2
import time
import signal
import mediapipe as mp

# Add core directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

from gesture_recognizer_setup import create_gesture_recognizer
from emotion_recognizer_setup import create_emotion_recognizer

# Tell Python to ignore terminal resize signals (SIGWINCH) so that the
# camera feed doesn't abruptly close
if hasattr(signal, 'SIGWINCH'):
    signal.signal(signal.SIGWINCH, signal.SIG_IGN)

gesture_recognizer = create_gesture_recognizer()
emotion_recognizer = create_emotion_recognizer()

# Get the video feed from the camera
# The camera has index 0 in Linux
cap = cv2.VideoCapture(0)

# Do hand gesture recognition every 200 milliseconds
last_gesture_infer_time = 0.0
gesture_interval = 200.0  # milliseconds
gesture_timestamp_ms = 0

# allow the camera feed window to be resized by the OS
has_gui = True
try:
    cv2.namedWindow('Camera Feed')
except Exception:
    has_gui = False

while True:
    success, frame = cap.read()
    if not success:
        # Sleep briefly if no camera frame is received to prevent busy waiting
        time.sleep(0.1)
        continue

    # Show the camera feed in an image
    if has_gui:
        try:
            cv2.imshow("Camera Feed", frame)
        except Exception:
            has_gui = False

    # OpenCV gives frames in BGR format. MediaPipe expects RGB.
    # Convert BGR -> RGB
    # cvtColor method converts an image from one color space to another
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Get a monotonically increasing timestamp to keep track of
    # time elapsed
    now = time.monotonic() * 1000
    now_ms = int(now)
    if now_ms <= gesture_timestamp_ms:
        now_ms = gesture_timestamp_ms + 1
    gesture_timestamp_ms = now_ms

    # Gesture recognition
    if now - last_gesture_infer_time >= gesture_interval:
        last_gesture_infer_time = now

        # Send to recognizer (once per 200 milliseconds)
        gesture_recognizer.recognize_async(mp_image, now_ms)

    # Emotion recognition
    emotion_recognizer.detect_async(mp_image, now_ms)

    if has_gui:
        try:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception:
            has_gui = False
    else:
        # Sleep slightly to prevent high CPU usage in headless mode
        time.sleep(0.01)

cap.release()
if has_gui:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
gesture_recognizer.close()
emotion_recognizer.close()
