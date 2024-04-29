import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from train_model import model

new_base_dir = pathlib.Path("dataset")

test_dataset = image_dataset_from_directory(
    new_base_dir / "test", image_size=(180, 180), batch_size=32
)

# model = keras.models.load_model("convnet_from_scratch.keras")

test_loss, test_accuracy = model.evaluate(test_dataset)

# print(f"Test loss: {test_loss}")
# print(f"Test accuracy: {test_accuracy}")

# Predict classes using the model on the test dataset
predictions = model.predict(test_dataset)
predicted_classes = np.argmax(predictions, axis=1)


# Retrieve true labels
true_labels = []
for images, labels in test_dataset:
    true_labels.extend(labels.numpy())
true_labels = np.array(true_labels)

# Output the classification report which includes precision, recall, f1-score, and accuracy per class
print(classification_report(true_labels, predicted_classes))

# Optionally display the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_classes)
print(conf_matrix)

# General test accuracy
# test_loss, test_accuracy = model.evaluate(test_dataset)
# print(f"Test loss: {test_loss}")
# print(f"Test accuracy: {test_accuracy}")

# Iterate over predictions and true labels to determine the group
for i in range(len(true_labels)):
    print(f"Test image {i+1} belongs to group {true_labels[i]}, predicted as group {predicted_classes[i]}")

