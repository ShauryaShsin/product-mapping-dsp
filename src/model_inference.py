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

start = "Start"

TEST_IMAGE = "src/dataset/test_2/pringles-original/ezgif-7-c2bd800053.jpg"

# Define the path to the test dataset
TEST_PATH = pathlib.Path("src/dataset/test_2/")

# List all subfolders within the main folder
TEST2_IMAGES_FOLDERS = [
    folder for folder in os.listdir(TEST_PATH) if (TEST_PATH / folder).is_dir()
]


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


#
# Loop through each subfolder
for folder in TEST2_IMAGES_FOLDERS:
    folder_path = TEST_PATH / folder
    # List all image files within the subfolder
    image_files = [
        file
        for file in os.listdir(folder_path)
        if file.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Loop through each image file and print its path
    for image in image_files:
        image_path = folder_path / image
        print("Product", folder.capitalize())
        print(f"Path: {image_path}")
        result = predict_product(img_path=image_path)
        print("Final prediction: ", result)
        print("Is correct:", result.capitalize() == folder.capitalize())
