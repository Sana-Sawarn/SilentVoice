import cv2
import pickle
import pyttsx3
import mediapipe as mp
import numpy as np
import time
from feature_extractor import extract_features
from dummy_model import DummyASLModel  # For loading dummy model

# Load the model
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

# TTS
engine = pyttsx3.init()

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not available.")
    exit()

# Prediction control
last_prediction_time = 0
prediction_interval = 2  # seconds
current_letter = ""
confirmed_text = ""

def is_fist(landmarks):
    """Check if hand is in a closed fist position using fingertip landmarks."""
    tips_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    fingers_folded = 0

    for tip_id in tips_ids[1:]:  # Skip thumb
        if landmarks[tip_id].y > landmarks[tip_id - 2].y:
            fingers_folded += 1

    # Optional: check thumb as well
    if landmarks[4].x < landmarks[3].x:  # Thumb folded across palm
        fingers_folded += 1

    return fingers_folded >= 4  # 4+ fingers folded = fist

print("✊ Show hand signs. Close fist to confirm the letter. Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Predict letter every few seconds
            current_time = time.time()
            if current_time - last_prediction_time >= prediction_interval:
                features = extract_features(handLms).reshape(1, -1)
                current_letter = model.predict(features)[0]
                last_prediction_time = current_time

            # Detect closed fist
            if is_fist(handLms.landmark):
                cv2.putText(frame, "✊ Fist Detected - Confirming", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                confirmed_text += current_letter
                print(f"✅ Confirmed Letter: {current_letter}")
                engine.say(current_letter)
                engine.runAndWait()
                time.sleep(1.5)  # Pause to avoid double-confirmation

    # Show prediction and confirmed text
    cv2.putText(frame, f"Current Prediction: {current_letter}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, f"Confirmed: {confirmed_text}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("SilentVoice - Fist Confirmation Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
