import cv2
import mediapipe as mp
from gesture_recognizer_setup import create_gesture_recognizer

recognizer = create_gesture_recognizer()

# Get the video feed from the camera
# The camera has index 0 in Linux
cap = cv2.VideoCapture(0)
timestamp = 0

# Gesture recognition for every frame
while True:
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR (OpenCV) -> RGB (MediaPipe)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Send to recognizer
    recognizer.recognize_async(mp_image, timestamp)
    timestamp += 33  # ~30 FPS

    cv2.imshow("Gesture Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
recognizer.close()
