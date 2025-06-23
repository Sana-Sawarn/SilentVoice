# collect_dynamic_gesture_data.py
import cv2
import mediapipe as mp
import csv
import os
import time
from feature_extractor import extract_features

GESTURES = ['hello', 'ok', 'stop', 'love', 'done']

# Create CSV file
with open('gesture_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'f{i}' for i in range(63)])  # 21 landmarks * 3 coords

    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    for gesture in GESTURES:
        print(f"\n✋ Show gesture: {gesture}")
        input("👉 Press Enter when ready...")

        count = 0
        while count < 100:  # Capture 100 frames per gesture
            success, frame = cap.read()
            if not success:
                continue

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    features = extract_features(handLms)
                    writer.writerow([gesture] + features.tolist())
                    count += 1
                    print(f"Captured {count}/100", end="\r")

            cv2.imshow("Collecting Gesture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
