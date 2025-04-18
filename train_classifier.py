import pickle  # Used to save and load Python objects (like models and datasets)
import numpy as np  # Library for handling numerical data efficiently
from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm for classification
from sklearn.model_selection import train_test_split  # Function to split dataset into training and testing sets
from sklearn.metrics import accuracy_score  # Function to calculate the accuracy of the model

# Load dataset from a saved pickle file
# 'rb' mode means 'read binary'
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert data into a NumPy array for efficient computation
# Ensure that the data is in floating-point format
# This step helps prevent datatype mismatch errors during training
data = np.asarray(data_dict['data'], dtype=np.float32)

# Convert labels into a NumPy array
# Ensure labels are stored as integers, as required by classification models
labels = np.asarray(data_dict['labels'], dtype=np.int32)

# Display basic information about the dataset
print(f"Loaded dataset with shape: {data.shape}")  # Show total samples and features
print(f"Sample data point: {data[0]}")  # Display an example feature vector
print(f"Sample label: {labels[0]}")  # Display corresponding label

# Split dataset into training and testing parts
# 80% of data is used for training, 20% for testing
# 'shuffle=True' ensures data is mixed before splitting
# 'stratify=labels' ensures the split maintains the same label distribution
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the Random Forest classifier
# n_estimators=100 -> Use 100 decision trees
# max_depth=10 -> Each tree can have a maximum depth of 10 (controls complexity)
# random_state=42 -> Ensures consistent results when running the code multiple times
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Train the classifier using the training data
model.fit(x_train, y_train)

# Use the trained model to predict labels for the test data
y_predict = model.predict(x_test)

# Calculate the accuracy by comparing predicted labels to actual labels
score = accuracy_score(y_predict, y_test)

# Display the model's accuracy in percentage format
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the trained model to a file for future use
# 'wb' mode means 'write binary'
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)  # Save model inside a dictionary
