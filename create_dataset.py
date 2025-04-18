import os  # To interact with the operating system (reading directories and files)
import pickle  # To save and load data efficiently
import mediapipe as mp  # To process hand landmarks
import cv2  # OpenCV for image processing
import numpy as np  # For handling numerical data

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands  # Load the Hands module from MediaPipe
hands = mp_hands.Hands(
    static_image_mode=True,  # Process each image independently
    min_detection_confidence=0.3  # Minimum confidence threshold for detection of the hand
)

# Path to the directory containing training images, organized by class
DATA_DIR = './data'

# Lists to store extracted feature data and corresponding labels
data = []  # Will store hand landmarks as feature vectors
labels = []  # Will store class labels corresponding to each feature vector

# Define a fixed feature size to maintain consistency across all samples
FEATURE_SIZE = 84  
# Number of landmarks per hand = 21

# Number of coordinates per landmark = 2 (x, y)

# Total features per hand = 21 × 2 = 42

# For both hands (left & right) = 42 × 2 = 84

# Loop through each folder (representing a class) inside the data directory
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)  # Full path to the class directory
    
    # Skip non-directory files (like .gitignore)
    if not os.path.isdir(dir_path):
        continue
    
    # Loop through each image in the class directory
    for img_path in os.listdir(dir_path):
        data_aux = []  # Temporary list to store processed landmark coordinates
        x_ = []  # List to store all x-coordinates for normalization
        y_ = []  # List to store all y-coordinates for normalization

        # Read the image
        img = cv2.imread(os.path.join(dir_path, img_path))
        
        # If the image cannot be read, skip it and show a warning
        if img is None:
            print(f"Warning: Failed to read {img_path}. Skipping...")
            continue
        
        # Convert image from BGR (OpenCV default) to RGB (required for MediaPipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image using MediaPipe to detect hand landmarks
        results = hands.process(img_rgb)
        
        # Check if any hands were detected in the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates of all landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x_.append(hand_landmarks.landmark[i].x)  # Store x-coordinate
                    y_.append(hand_landmarks.landmark[i].y)  # Store y-coordinate
                
                # Normalize landmarks by shifting all points relative to the minimum x and y values
                for i in range(len(hand_landmarks.landmark)):
                    data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                    data_aux.append(hand_landmarks.landmark[i].y - min(y_))
            
            # Ensure each feature vector has exactly FEATURE_SIZE values
            while len(data_aux) < FEATURE_SIZE:
                data_aux.append(0)  # Pad with zeros if there are fewer points
            
            # If more points exist (shouldn't happen), trim the extra values
            data_aux = data_aux[:FEATURE_SIZE]
            
            # Add the processed feature vector to the dataset
            data.append(data_aux)
            labels.append(int(dir_))  # Convert directory name (class label) to an integer

# Convert feature data and labels into NumPy arrays for better performance
data = np.array(data, dtype=np.float32)  # Ensure floating-point values for model compatibility
labels = np.array(labels, dtype=np.int32)  # Store labels as integers

# Save the dataset into a pickle file for easy loading later
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Print summary of dataset creation
print(f"Dataset successfully created with {len(data)} samples.")
