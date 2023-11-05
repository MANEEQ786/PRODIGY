import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2
import os


dataset_path = 'asl_dataset'

# Define the list of gestures (0 to 5)
gestures = ['0', '1', '2', '3', '4', '5']

# Function to preprocess an image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    return image

# Load and preprocess the dataset
images = []
labels = []

for gesture in gestures:
    gesture_path = os.path.join(dataset_path, gesture)
    for filename in os.listdir(gesture_path):
        if filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(gesture_path, filename))
            image = preprocess_image(image)
            images.append(image)
            labels.append(int(gesture))

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalize pixel values to [0, 1]
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define the CNN model
model = keras.Sequential([
    layers.Input(shape=(64, 64)),
    layers.Reshape((64, 64, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy (image dataset): {test_accuracy * 100:.2f}%')

# Save the trained model for future use
model.save('asl_gesture_recognition_model.h5')

# capture video frames and recognize gestures
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the captured frame
    processed_frame = preprocess_image(frame)
    processed_frame = np.expand_dims(processed_frame, axis=0) / 255.0

    # Make a prediction
    prediction = model.predict(processed_frame)
    gesture = gestures[np.argmax(prediction)]

    # Display the recognized gesture
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
