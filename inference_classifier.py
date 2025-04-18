import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
import subprocess
import pyautogui
import time
import pyttsx3
import threading

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Webcam init
cap = cv2.VideoCapture(0)

# MediaPipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3
)

labels_dict = {0: 'H', 1: 'E', 2: 'L', 3: 'O'}

# TTS Engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    if "female" in voice.name.lower() or "zira" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break
engine.setProperty('rate', 180)
engine.setProperty('volume', 1.0)
pronunciations = {"H": "H", "E": "E", "L": "L", "O": "O"}

def speak(text):
    spoken_text = pronunciations.get(text, text)
    threading.Thread(target=lambda: (engine.say(spoken_text), engine.runAndWait())).start()

# Open Notepad
subprocess.Popen("notepad.exe")
time.sleep(1)

recognized_text = ""
last_detected_character = ""
detection_start_time = None
repeat_start_time = None
confirmation_time = 1.5  # seconds to confirm a new character
repeat_delay = 1.5       # seconds between repeated characters if same sign is held

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Could not capture frame.")
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # MODIFIED: Create a blurred version of the frame
    blurred_frame = cv2.GaussianBlur(frame, (99, 99), 0)
    output_frame = blurred_frame.copy()

    predicted_character = ""
    probability = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on original frame for later pasting
        frame_with_hand = frame.copy()
        mp_drawing.draw_landmarks(
            frame_with_hand, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        for lm in hand_landmarks.landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        while len(data_aux) < 84:
            data_aux.append(0)
        data_aux = data_aux[:84]

        features = np.asarray(data_aux).reshape(1, -1)
        probabilities = model.predict_proba(features)[0]
        prediction_index = np.argmax(probabilities)
        predicted_character = labels_dict[prediction_index]
        probability = probabilities[prediction_index]

        current_time = time.time()

        if predicted_character != last_detected_character:
            detection_start_time = current_time
            last_detected_character = predicted_character
            repeat_start_time = None

        if detection_start_time and (current_time - detection_start_time) >= confirmation_time:
            if not recognized_text or recognized_text[-1] != predicted_character:
                recognized_text += predicted_character
                pyautogui.typewrite(predicted_character)
                speak(predicted_character)
                detection_start_time = None
                repeat_start_time = current_time

        elif repeat_start_time and (current_time - repeat_start_time) >= repeat_delay:
            recognized_text += predicted_character
            pyautogui.typewrite(predicted_character)
            speak(predicted_character)
            repeat_start_time = current_time

        # MODIFIED: Define hand bounding box
        x_min = max(int(min(x_) * W) - 20, 0)
        x_max = min(int(max(x_) * W) + 20, W)
        y_min = max(int(min(y_) * H) - 20, 0)
        y_max = min(int(max(y_) * H) + 20, H)

        # MODIFIED: Copy the hand region from the original frame to the blurred one
        output_frame[y_min:y_max, x_min:x_max] = frame_with_hand[y_min:y_max, x_min:x_max]

        # Add label and confidence
        cv2.putText(output_frame, f'{predicted_character} ({int(probability * 100)}%)',
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # MODIFIED: Show the modified output frame
    cv2.imshow('Sign Language Detector', output_frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 8:  # Backspace
        recognized_text = recognized_text[:-1]
        pyautogui.hotkey('backspace')
    elif key == ord('c'):  # Clear
        recognized_text = ""
        pyautogui.hotkey('ctrl', 'a')
        pyautogui.hotkey('backspace')

cap.release()
cv2.destroyAllWindows()
