import cv2
import time
import signal
import mediapipe as mp
from gesture_recognizer_setup import create_gesture_recognizer
from emotion_recognizer_setup import create_emotion_recognizer

# Tell Python to ignore terminal resize signals (SIGWINCH) so that the
# camera feed doesn't abruptly close
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
cv2.namedWindow('Camera Feed')

while True:
    success, frame = cap.read()
    if not success:
        break

    # Show the camera feed in an image
    cv2.imshow("Camera Feed", frame)

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
        gesture_recognizer.recognize_async(mp_image, int(now))

    # Emotion recognition
    emotion_recognizer.detect_async(mp_image, int(now))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
gesture_recognizer.close()
emotion_recognizer.close()
