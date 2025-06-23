import cv2
import pickle
import pyttsx3
import mediapipe as mp
import numpy as np
import time
import os
from feature_extractor import extract_features

# === Load Gesture Classifier ===
gesture_model = None
if os.path.exists("gesture_classifier.pkl"):
    with open("gesture_classifier.pkl", "rb") as f:
        gesture_model = pickle.load(f)
        print("✅ Gesture classifier loaded.")
else:
    print("⚠️ Warning: gesture_classifier.pkl not found!")

# === TTS Setup ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === MediaPipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# === Camera Setup ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit()
else:
    print("✅ Camera opened.")

# === Mode Selection ===
mode = input("Select mode (letter/word/phrase/gesture): ").strip().lower()
if mode == "word":
    num_letters = int(input("How many letters in the word? "))
    full_result = ""
    letters_remaining = num_letters
elif mode == "phrase":
    num_words = int(input("How many words in the phrase? "))
    phrase_result = []
    current_word = ""
    words_remaining = num_words
elif mode == "letter":
    full_result = ""
elif mode == "gesture":
    if gesture_model is None:
        print("❌ Gesture mode requires a trained gesture_classifier.pkl")
        cap.release()
        exit()
else:
    print("❌ Invalid mode.")
    cap.release()
    exit()

# === A–Z Setup ===
letters = [chr(i) for i in range(65, 91)]
letter_index = 0
current_letter = letters[letter_index]
last_switch_time = time.time()
auto_scroll = True
scroll_direction = 1
last_confirm_time = 0
last_spoken = ""
gesture_delay = 2

# === Fist Detection ===
def is_fist(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    folded = 0
    for i in range(1, 5):
        if landmarks[tips_ids[i]].y > landmarks[tips_ids[i] - 2].y:
            folded += 1
    if landmarks[4].x < landmarks[3].x:
        folded += 1
    return folded >= 4

print("\n🔁 Press 'c' to continue scrolling letters")
print("🔁 Press 'r' to reverse letter scroll direction")
print("⌫ Press 'd' to delete")
print("➡️ Press 'n' for next word (phrase mode)")
print("🗑 Press 'x' to reset")
print("❌ Press 'q' to quit\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hands = hands.process(img_rgb)

    # Letter scrolling (not in gesture mode)
    if mode != "gesture" and auto_scroll and time.time() - last_switch_time >= 1.0:
        letter_index = (letter_index + scroll_direction) % len(letters)
        current_letter = letters[letter_index]
        last_switch_time = time.time()

    if result_hands.multi_hand_landmarks:
        for handLms in result_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            features = extract_features(handLms).reshape(1, -1)

            if mode == "gesture":
                prediction = gesture_model.predict(features)[0]
                cv2.putText(frame, f"Gesture: {prediction}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

                if prediction != last_spoken or time.time() - last_confirm_time > gesture_delay:
                    print(f"🗣️ {prediction}")
                    engine.say(prediction)
                    engine.runAndWait()
                    last_spoken = prediction
                    last_confirm_time = time.time()

            elif is_fist(handLms.landmark) and time.time() - last_confirm_time > 1.5:
                print(f"✊ Confirmed: {current_letter}")
                engine.say(current_letter)
                engine.runAndWait()
                last_confirm_time = time.time()
                auto_scroll = False

                if mode == "letter":
                    full_result += current_letter

                elif mode == "word":
                    full_result += current_letter
                    letters_remaining -= 1
                    if letters_remaining == 0:
                        print(f"✅ Final Word: {full_result}")
                        engine.say("Final word is " + full_result)
                        engine.runAndWait()
                        time.sleep(2)
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                elif mode == "phrase":
                    current_word += current_letter

    # === Display
    if mode != "gesture":
        cv2.putText(frame, f"Current: {current_letter}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    if mode == "word":
        cv2.putText(frame, f"Word: {full_result}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    elif mode == "letter":
        cv2.putText(frame, f"Result: {full_result}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    elif mode == "phrase":
        phrase_str = ' '.join(phrase_result + ([current_word] if current_word else []))
        cv2.putText(frame, f"Phrase: {phrase_str}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

    cv2.imshow("SilentVoice - Full Version", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        auto_scroll = True
        last_switch_time = time.time()
    elif key == ord('r'):
        scroll_direction *= -1
    elif key == ord('d'):
        if mode == "word" and full_result:
            full_result = full_result[:-1]
            letters_remaining += 1
        elif mode == "letter" and full_result:
            full_result = full_result[:-1]
        elif mode == "phrase" and current_word:
            current_word = current_word[:-1]
    elif key == ord('n') and mode == "phrase":
        if current_word:
            phrase_result.append(current_word)
            words_remaining -= 1
            current_word = ""
            if words_remaining == 0:
                final_phrase = ' '.join(phrase_result)
                print(f"✅ Final Phrase: {final_phrase}")
                engine.say("Final phrase is " + final_phrase)
                engine.runAndWait()
                time.sleep(2)
                cap.release()
                cv2.destroyAllWindows()
                exit()
    elif key == ord('x'):
        if mode == "word":
            full_result = ""
            letters_remaining = num_letters
        elif mode == "letter":
            full_result = ""
        elif mode == "phrase":
            phrase_result = []
            current_word = ""
            words_remaining = num_words

cap.release()
cv2.destroyAllWindows()
