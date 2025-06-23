import cv2
import mediapipe as mp
import csv
import os

GESTURES = ["hello", "stop", "ok", "love", "done"]
DATA_FILE = "gesture_dataset.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not available")
    exit()

print("\n▶ Press keys 1–5 to record gestures:")
for i, g in enumerate(GESTURES):
    print(f"  {i+1}: {g}")
print("💾 Press 's' to save dataset")
print("❌ Press 'q' to quit\n")

all_data = []

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Collector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in [ord(str(i+1)) for i in range(len(GESTURES))]:
        gesture = GESTURES[int(chr(key)) - 1]
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                row = [gesture]
                for lm in handLms.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                all_data.append(row)
                print(f"✅ Captured: {gesture}")

    elif key == ord('s'):
        with open(DATA_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"f{i}" for i in range(len(all_data[0])-1)]
            writer.writerow(header)
            writer.writerows(all_data)
        print(f"💾 Saved to {DATA_FILE}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
