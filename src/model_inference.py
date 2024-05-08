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
from config import IMAGE_SIZE

TEST_IMAGE = "src/dataset/test/kims-havsalt/rema_image.jpg"


def predict_product(img_path: str) -> str:
    """Predicts product class using validation folder"""
    image_size = IMAGE_SIZE

    # Setup paths
    dir = pathlib.Path("src/dataset")
    dataset = image_dataset_from_directory(
        dir / "test", image_size=image_size, batch_size=16
    )

    # Load model
    model = keras.models.load_model("src/models/convnet_from_scratch.keras")

    img = keras.utils.load_img(img_path, target_size=image_size)
    # plt.imshow(img)

    # Store image in numpy array
    img_array = keras.utils.img_to_array(img)
    img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

    predictions = model.predict(img_array)

    # Apply softmax to convert logits to probabilities
    probabilities = keras.activations.softmax(predictions[0]).numpy()

    # Get classes from dataset
    classes = (
        dataset.class_names
    )  # classes = ["favorit", "havsalt", "sour"]  # Modify as per your actual classes

    # Print the probabilities for each class
    for class_name, probability in zip(classes, probabilities):
        print(f"This image is {probability * 100:.2f}% {class_name}.")

    return classes[np.argmax(probabilities)]


print("Predicted product is: ", predict_product(img_path=TEST_IMAGE))
