import cv2
import time
import mediapipe as mp
from gesture_recognizer_setup import create_gesture_recognizer

recognizer = create_gesture_recognizer()

# Get the video feed from the camera
# The camera has index 0 in Linux
cap = cv2.VideoCapture(0)
timestamp = 0

# Do hand gesture recognition every second, not every frame

last_infer_time = 0.0
interval = 1.0  # seconds

timestamp_ms = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Show the camera feed in an image
    cv2.imshow("Gesture Stream", frame)

    # Get a monotonically increasing timestamp to keep track of
    # time 8elapsed
    now = time.monotonic()
    if now - last_infer_time >= interval:
        last_infer_time = now

        # OpenCV gives frames in BGR format. MediaPipe expects RGB.
        # Convert BGR -> RGB
        # cvtColor method converts an image from one color space to another
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Send to recognizer (once per second)
        recognizer.recognize_async(mp_image, timestamp_ms)
        timestamp_ms += int(interval * 1000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()
