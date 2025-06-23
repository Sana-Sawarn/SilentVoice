import cv2
import mediapipe as mp
import pyttsx3
import time

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not available.")
    exit()

# Choose mode
mode = input("Enter mode (word/phrase/gesture): ").strip().lower()

if mode == "word":
    num_letters = int(input("How many letters in the word? "))
    letters_remaining = num_letters
    full_result = ""
elif mode == "phrase":
    num_words = int(input("How many words in the phrase? "))
    current_word = ""
    phrase_result = []
    words_remaining = num_words
else:
    print("❌ Invalid mode")
    exit()

# A–Z setup
letters = [chr(i) for i in range(65, 91)]
letter_index = 0
current_letter = letters[letter_index]
last_switch_time = time.time()
scroll_direction = 1  # +1 for A→Z, -1 for Z→A

# Control variables
auto_scroll = True
last_confirm_time = 0

# Fist detection function
def is_fist(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    folded = 0
    for i in range(1, 5):
        if landmarks[tips_ids[i]].y > landmarks[tips_ids[i] - 2].y:
            folded += 1
    if landmarks[4].x < landmarks[3].x:
        folded += 1
    return folded >= 4

print("\n🔁 Letters loop A–Z every 1 sec (default)")
print("✊ Fist = confirm letter (freezes)")
print("➡️ Press 'c' to continue to next letter (auto-scroll resumes)")
print("🔁 Press 'r' to toggle scroll direction (A→Z / Z→A)")
print("⌫ Press 'd' to delete last letter")
print("➡️ Press 'n' to go to next word (in phrase mode)")
print("🗑 Press 'x' to reset word/phrase")
print("❌ Press 'q' to quit\n")

while True:
    success, frame = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # Auto-scroll every 1 second in current direction
    if auto_scroll and time.time() - last_switch_time >= 1.0:
        letter_index = (letter_index + scroll_direction) % len(letters)
        current_letter = letters[letter_index]
        last_switch_time = time.time()

    # Fist detection
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            if is_fist(handLms.landmark) and time.time() - last_confirm_time > 1.5:
                print(f"✅ Confirmed: {current_letter}")
                engine.say(current_letter)
                engine.runAndWait()
                last_confirm_time = time.time()
                auto_scroll = False  # Freeze letter

                if mode == "word":
                    full_result += current_letter
                    letters_remaining -= 1
                    if letters_remaining == 0:
                        print("✅ Final Word:", full_result)
                        engine.say("Final word is " + full_result)
                        engine.runAndWait()
                        time.sleep(2)
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                elif mode == "phrase":
                    current_word += current_letter

    # Display text
    cv2.putText(frame, f"Current: {current_letter}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    if mode == "word":
        cv2.putText(frame, f"Word: {full_result}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        cv2.putText(frame, f"Remaining: {letters_remaining}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)
    elif mode == "phrase":
        phrase_str = ' '.join(phrase_result + ([current_word] if current_word else []))
        cv2.putText(frame, f"Phrase: {phrase_str}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        cv2.putText(frame, f"Words Left: {words_remaining}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)

    cv2.imshow("SilentVoice - Bidirectional Scroll", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('c'):
        auto_scroll = True
        last_switch_time = time.time()
    elif key == ord('r'):
        scroll_direction *= -1
        print("🔄 Direction reversed!" if scroll_direction == -1 else "🔁 Direction forward!")
    elif key == ord('d'):
        if mode == "word" and full_result:
            full_result = full_result[:-1]
            letters_remaining += 1
            print("❌ Deleted last letter")
        elif mode == "phrase" and current_word:
            current_word = current_word[:-1]
            print("❌ Deleted last letter from current word")
    elif key == ord('n') and mode == "phrase":
        if current_word:
            phrase_result.append(current_word)
            words_remaining -= 1
            current_word = ""
            if words_remaining == 0:
                final_phrase = ' '.join(phrase_result)
                print("✅ Final Phrase:", final_phrase)
                engine.say("Final phrase is " + final_phrase)
                engine.runAndWait()
                time.sleep(2)
                cap.release()
                cv2.destroyAllWindows()
                exit()
    elif key == ord('x'):
        # Reset word or phrase
        if mode == "word":
            full_result = ""
            letters_remaining = num_letters
            print("🗑 Word reset.")
        elif mode == "phrase":
            current_word = ""
            phrase_result = []
            words_remaining = num_words
            print("🗑 Phrase reset.")

cap.release()
cv2.destroyAllWindows()
