"""
Use the trained model for inference.
Ressources:
https://keras.io/examples/vision/image_classification_from_scratch/
"""

import os, shutil, pathlib
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory


def predict_all_products(test_dir: str):
    """Predicts product class for all images in the test directory"""
    # Setup paths
    test_dataset = image_dataset_from_directory(
        test_dir, image_size=(180, 180), batch_size=16
    )

    # Load model
    model = keras.models.load_model("src/models/convnet_from_scratch.keras")

    # Get classes from dataset
    classes = test_dataset.class_names

    # Predict for each image in the test dataset
    for images, labels in test_dataset:
        predictions = model.predict(images)
        for i in range(len(labels)):
            folder_name = os.path.basename(os.path.dirname(test_dataset.file_paths[i]))
            probabilities = keras.activations.softmax(predictions[i]).numpy()
            predicted_class_index = np.argmax(probabilities)
            predicted_class = classes[predicted_class_index]
            print(f"Folder: {folder_name}, Predicted class: {predicted_class}")


test_dir = pathlib.Path("src/dataset/test")
predict_all_products(test_dir)
