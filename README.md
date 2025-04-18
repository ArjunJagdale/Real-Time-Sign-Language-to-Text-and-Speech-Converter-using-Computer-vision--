# Real-Time Sign Language to Text and Speech Converter

This project is a **real-time Sign Language to Text and Speech Converter** using hand gesture recognition. It captures hand gestures using a webcam, classifies them into corresponding letters (H, E, L, O), and converts them to text and speech in real-time. The project leverages **MediaPipe** for hand gesture detection and **OpenCV** for image processing, with a **Random Forest** classifier trained to recognize the gestures.

## Demo


https://github.com/user-attachments/assets/37b8eac7-abd7-4102-8ecd-e00f2314d5ac



## Project Overview

- **Real-Time Gesture Recognition**: Captures and classifies hand gestures into corresponding sign language characters (H, E, L, O).
- **Text and Speech Output**: Converts recognized hand signs into text and speaks the corresponding letter.
- **Custom Dataset**: Collected a custom dataset with 100 images per class using webcam input.
- **Technologies Used**: 
  - Python
  - MediaPipe
  - OpenCV
  - Random Forest Classifier
  - Text-to-Speech (pyttsx3)
  - Notepad automation (pyautogui)

## Features

- **Real-time Gesture Detection**: Uses a webcam to capture hand gestures and classify them into letters.
- **Accuracy**: A Random Forest model is trained to classify hand gestures with high accuracy.
- **Text and Speech Conversion**: Converts the recognized sign language gestures into text output and generates speech using pyttsx3.
- **Automatic Notepad Typing**: Recognized letters are typed into a Notepad in real-time for easy communication.

## Getting Started

### Prerequisites

To run this project, ensure you have Python installed on your system. You also need to install the following dependencies:

```bash
pip install mediapipe opencv-python pyttsx3 scikit-learn numpy pyautogui
```

### Dataset

The dataset for this project consists of hand gesture images for each of the characters **H**, **E**, **L**, and **O**. A total of 100 images were captured per class using the webcam. The features for each image were extracted using MediaPipe’s hand landmarks.

### Training the Classifier

The Random Forest classifier is trained on the extracted landmark features to classify the hand gestures. After training, the model is saved to a file using Pickle.

```python
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load features and labels
features = ...  # Load preprocessed features
labels = ...    # Load corresponding labels

# Train the Random Forest model
clf = RandomForestClassifier()
clf.fit(features, labels)

# Save the trained model
with open("hand_gesture_model.pkl", "wb") as file:
    pickle.dump(clf, file)
```

### Real-Time Gesture Recognition and Text/Speech Conversion

To run the real-time gesture recognition system, execute the following script:

```bash
python collect_imgs.py
python create_dataset.py
python train_classifier.py
python inference_classifier.py
```

This will activate the webcam, start detecting hand gestures, and perform the corresponding actions:
1. Convert the gesture to text.
2. Speak the letter using **pyttsx3**.
3. Type the recognized letter into Notepad using **pyautogui**.

### Example:

When you sign the letter "H", the program will output:

- Text: "H"
- Speech: "H"
- Automatically type "H" into the Notepad.

## How It Works

1. **Gesture Detection**: MediaPipe's hand landmarks are used to detect the position of hand joints.
2. **Feature Extraction**: The hand landmarks (84 features) are extracted for each gesture.
3. **Classification**: The Random Forest model classifies the gesture based on the extracted features.
4. **Text & Speech Output**: Once classified, the recognized letter is spoken and displayed in real-time in Notepad.

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/) for the hand gesture detection.
- [OpenCV](https://opencv.org/) for image processing.
- [pyttsx3](https://pypi.org/project/pyttsx3/) for text-to-speech functionality.
- [scikit-learn](https://scikit-learn.org/) for machine learning classification.
```
