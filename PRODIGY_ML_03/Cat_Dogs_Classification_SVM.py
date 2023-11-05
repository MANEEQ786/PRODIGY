# Import Libraries
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import seaborn as sns


# Set the path to the training data folder
train_data_path = 'TRAIN'

#Visualize images of cat and dog
img1 = 'cat.3.jpg'
img2 = 'dog.13.jpg'
image_file1 = os.path.join(train_data_path, img1)
image_file2 = os.path.join(train_data_path, img2)
img = cv2.imread(image_file1)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Cat Image')
img = cv2.imread(image_file2)
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Dog Image')
plt.tight_layout()
plt.show()

# Initialize empty lists to store images and labels
images = []
labels = []

# Load and preprocess training data
for image_file in os.listdir(train_data_path):
    if image_file.endswith(".jpg"):
        label = "cat" if "cat" in image_file else "dog"
        image = cv2.imread(os.path.join(train_data_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (128, 128))  # Resize the image
        images.append(image)
        labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Flatten the images and reshape
images = images.reshape(images.shape[0], -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create and train an SVM model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Set the path to the testing data folder
test_data_path = 'TEST'

# Initialize empty lists to store test images
test_images = []

# Load and preprocess testing data
for image_file in os.listdir(test_data_path):
    if image_file.endswith(".jpg"):  # Filter only image files
        image = cv2.imread(os.path.join(test_data_path, image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        image = cv2.resize(image, (128, 128))  # Resize the image
        test_images.append(image)

# Convert the test images to a numpy array
test_images = np.array(test_images)

# Flatten the test images and reshape
test_images = test_images.reshape(test_images.shape[0], -1)

# Make predictions on the test set
y_pred = svm_model.predict(test_images)

# Calculate accuracy and create a confusion matrix
accuracy = accuracy_score(y_test, svm_model.predict(X_test))*100
report = classification_report(y_test, svm_model.predict(X_test))

confusion = confusion_matrix(y_test, svm_model.predict(X_test))

# Visualize the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues', xticklabels=['cat', 'dog'], yticklabels=['cat', 'dog'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Accuracy: {accuracy:.2f}')
plt.show()

print(f"Accuracy: {accuracy}")
print(report)


submission_data = {
    "id": [i for i in range(1, len(test_images) + 1)],
    "label": y_pred
}

# Create a CSV file for submission

submission_df = pd.DataFrame(submission_data)
submission_df.to_csv('sampleSubmission.csv', index=False)

print("Submission file 'submission.csv' has been created.")