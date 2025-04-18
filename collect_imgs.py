import os  # Import the os module to interact with the operating system (e.g., creating folders)
import cv2  # Import OpenCV, a library for image and video processing

# Define the directory where the dataset will be stored
DATA_DIR = './data'

# Check if the data directory exists; if not, create it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Set the number of classes (categories) for data collection
number_of_classes = 4  # We are collecting data for 4 different categories

# Define how many images we want to collect per class
dataset_size = 100  # Each class will have 100 images

# Initialize the webcam (0 represents the default webcam)
cap = cv2.VideoCapture(0)

# Loop through each class to collect images
for j in range(number_of_classes):
    # Create a folder for each class inside the data directory
    class_dir = os.path.join(DATA_DIR, str(j))  # Path for the class directory
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)  # Create the class directory if it does not exist

    print(f'Collecting data for class {j}')  # Print message to indicate which class is being recorded

    # Display instructions on the screen before starting data collection
    while True:
        ret, frame = cap.read()  # Capture a frame from the webcam
        if not ret:
            break  # If the frame is not captured correctly, exit the loop
        
        # Display a message on the video feed asking the user to press 'Q' to start
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame in a window named 'frame'
        
        # If the user presses 'Q', exit this loop and start capturing images
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0  # Counter to keep track of the number of images saved
    
    # Capture and save images for the current class
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a new frame from the webcam
        if not ret:
            break  # If the frame is not captured correctly, exit the loop
        
        cv2.imshow('frame', frame)  # Display the frame on the screen
        cv2.waitKey(25)  # Short delay to control the capture speed
        
        # Save the captured frame as an image file in the respective class folder
        image_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(image_path, frame)
        
        counter += 1  # Increase the counter after saving an image

# Release the webcam resource after all images are collected
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
