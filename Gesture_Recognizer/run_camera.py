import cv2
import time
import queue
import mediapipe as mp
from gesture_recognizer_setup import create_gesture_recognizer
from emotion_recognizer_setup import create_emotion_recognizer

gesture_recognizer = create_gesture_recognizer()
emotion_recognizer = create_emotion_recognizer()

# Get the video feed from the camera
# The camera has index 0 in Linux
cap = cv2.VideoCapture(0)

# Do hand gesture recognition every 200 milliseconds
last_gesture_infer_time = 0.0
gesture_interval = 200.0  # milliseconds
gesture_timestamp_ms = 0

# Do facial emotion recognition every 2 seconds
last_emotion_infer_time = 0.0
emotion_interval = 2000.0 # milliseconds
emotion_timestamp_ms = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Show the camera feed in an image
    cv2.imshow("Gesture Stream", frame)

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

    # Gesture recognition
    if now - last_gesture_infer_time >= gesture_interval:
        last_gesture_infer_time = now

        # Send to recognizer (once per 200 milliseconds)
        gesture_recognizer.recognize_async(mp_image, gesture_timestamp_ms)
        gesture_timestamp_ms += int(gesture_interval)

    # Emotion recognition
    if now - last_emotion_infer_time >= emotion_interval:
        last_emotion_infer_time = now

        # Send to recognizer (once per 2 seconds)
        emotion_recognizer.detect_async(mp_image, emotion_timestamp_ms)
        emotion_timestamp_ms += int(emotion_interval)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
gesture_recognizer.close()
