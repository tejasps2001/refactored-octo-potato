import os
import cv2
import sys
import mediapipe as mp

# Add the parent directory to the path so we can import the setup scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emotion_recognizer_setup import create_emotion_recognizer, predict_emotion, VisionRunningMode

def test_frustration_images():
    # Create the recognizer in IMAGE mode
    recognizer = create_emotion_recognizer(running_mode=VisionRunningMode.IMAGE)

    test_folder = os.path.dirname(os.path.abspath(__file__))
    frustration_folder = os.path.join(test_folder, "Emotion_Recognition", "Frustrated")
    images_count = len(os.listdir(frustration_folder))
    correct_guess_counter = 0
    incorrect_guess_image_path_list = []

    for filename in os.listdir(frustration_folder):
        if filename.endswith((".jpg", ".png", "jpeg", ".avif")):
            image_path = os.path.join(frustration_folder, filename)

            # Load and convert image for MediaPipe
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            result = recognizer.detect(mp_image)

            if result.face_blendshapes and result.face_blendshapes[0]:
                prediction = predict_emotion(result)
                print("PREDICTION IS", prediction)
                if prediction == "is frustrated":
                    correct_guess_counter += 1
                else:
                    incorrect_guess_image_path_list.append(image_path)
            else:
                print(f"Image: {filename} | No face detected.")

    # If at least one prediction is made, then output the test results
    if prediction:
        print(f"Guessed {correct_guess_counter} images correctly out of {images_count}.")
        print(f"Accuracy: {correct_guess_counter/images_count * 100}%")      
    else:
        print("No faces detected.")          

if __name__ == "__main__":
    test_frustration_images()