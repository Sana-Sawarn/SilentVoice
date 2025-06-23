import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from feature_extractor import extract_features

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Setup camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access camera.")
    exit()

# CSV data list
data = []
labels = []

print("🔡 Press a key A–Z to start recording that letter.")
print("⏹ Press 's' to stop recording that letter.")
print("💾 Press 'q' to save and exit.")

current_label = None
recording = False

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if recording and current_label is not None:
                features = extract_features(handLms)
                data.append(features)
                labels.append(current_label)
                cv2.putText(frame, f"Recording: {current_label}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Collect Hand Sign Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        recording = False
        print(f"🛑 Stopped recording for: {current_label}")
        current_label = None
    elif 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
        current_label = chr(key).upper()
        recording = True
        print(f"🎬 Started recording for: {current_label}")

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels
df.to_csv("hand_data.csv", index=False)
print("✅ Saved hand_data.csv")
