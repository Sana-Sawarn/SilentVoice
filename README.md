# 🧏‍♀️ SilentVoice – AI-Powered Sign Language to Speech Converter

**SilentVoice** is a real-time hand sign and gesture recognition system that helps non-verbal individuals communicate using intuitive hand motions. The system uses MediaPipe for hand tracking, machine learning for sign classification, and a text-to-speech engine to vocalize detected letters, words, phrases, or gestures.

---

## 🔥 Features

- 📷 Real-time sign language recognition (A–Z letters)
- 🧠 ML-based gesture recognition (OK, Hello, Done, Stop, Love, etc.)
- ✊ Fist-based letter confirmation
- 🔁 Auto-scrolling letter loop with bidirectional control
- 📢 Offline voice output using `pyttsx3`
- 🧠 Gesture classification powered by scikit-learn
- 🧼 Reset, delete, next word/phrase controls
- 👨‍🦽 Built to aid non-verbal communication

---

## 🎮 Modes

### 1. **Letter Mode (Word)**
- Auto-loops A–Z every 1 second.
- User confirms a letter by closing their **fist**.
- System asks for word length in letters (e.g., 5).
- Word is spoken aloud when completed.

### 2. **Phrase Mode**
- Collects multiple words one at a time.
- Uses same fist-confirm for letter.
- You confirm when a word is done and move to the next.
- Final phrase is spoken aloud.

### 3. **Gesture Mode**
- Detects full-hand gestures like:
  - ✋ `Hello`
  - 👍 `OK`
  - 🛑 `Stop`
  - 🤟 `Love`
  - ✅ `Done`
- Model trained using custom data with MediaPipe features.
- Voice output for recognized gesture.

---

## 🧠 Technologies Used

| Tool         | Purpose                              |
|--------------|--------------------------------------|
| MediaPipe    | Real-time hand landmark detection    |
| OpenCV       | Camera feed & visual overlay         |
| pyttsx3      | Offline text-to-speech engine        |
| scikit-learn | Machine learning gesture classifier  |
| NumPy        | Feature handling and processing      |

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Sana-Sawarn/SilentVoice.git
cd SilentVoice
