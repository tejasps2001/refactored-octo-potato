import cv2
import time
import mediapipe as mp
from gesture_recognizer_setup import create_gesture_recognizer

recognizer = create_gesture_recognizer()

# Get the video feed from the camera
# The camera has index 0 in Linux
cap = cv2.VideoCapture(0)

# Do hand gesture recognition every 200 milliseconds
last_gesture_infer_time = 0.0
gesture_interval = 200.0  # milliseconds
gesture_timestamp_ms = 0



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
    if now - last_gesture_infer_time >= gesture_interval:
        last_gesture_infer_time = now

        # Send to recognizer (once per 200 milliseconds)
        recognizer.recognize_async(mp_image, gesture_timestamp_ms)
        gesture_timestamp_ms += int(gesture_interval)

    # if now - last_

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()
