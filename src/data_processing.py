"""Processess and prepares images for model training"""

import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

# TODO: Rema images should automatically be put into the test set

CLEANUP_DIRS = True
NUMBER_OF_VARIATIONS = 80
NUMBER_OF_VARIATIONS_val = 20

directory_input = "src/product_images"
directory_output = "src/dataset"

product_classes = [
    filename for filename in os.listdir(directory_input) if filename != ".DS_Store"
]

print(product_classes)

# Clean up dirs
if CLEANUP_DIRS is True:
    if os.path.exists("src/dataset/validation"):
        shutil.rmtree("src/dataset/validation")
    if os.path.exists("src/dataset/train"):
        shutil.rmtree("src/dataset/train")
    # if os.path.exists("src/dataset/test"):
    #     shutil.rmtree("src/dataset/test")

for fileclass in product_classes:
    # Loop through the files in the directory
    # Construct full file path
    filepath = os.path.join(directory_input, fileclass, "image.jpg")

    # Load your image
    image = load_img(filepath)  # Make sure to provide the path to your image
    image = img_to_array(image)  # Convert the image to numpy array
    image = np.expand_dims(
        image, axis=0
    )  # Add a new axis to make the image array four dimensional

    # Copy original images to validation directory
    validation_dir = f"{directory_output}/validation/{fileclass}"
    os.makedirs(validation_dir, exist_ok=True)
    shutil.copyfile(filepath, validation_dir + "/image.jpg")

    # Create augmentations from the original images into the train dataset
    for type in ["train"]:
        save_dir = f"{directory_output}/{type}/{fileclass}/"
        os.makedirs(save_dir, exist_ok=True)

        # Create an ImageDataGenerator for data augmentation
        datagen = ImageDataGenerator(
            rotation_range=40,  # Degree range for random rotations
            width_shift_range=0.2,  # Range (as a fraction of total width) for horizontal shifts
            height_shift_range=0.2,  # Range (as a fraction of total height) for vertical shifts
            shear_range=0.2,  # Shearing intensity (shear angle in counter-clockwise direction)
            zoom_range=0.2,  # Range for random zoom
            # horizontal_flip=True,  # Randomly flip inputs horizontally
            fill_mode="nearest",  # Strategy to fill newly created pixels
        )

        # Generate batches of augmented images and save them to the specified directory
        augmented_images = datagen.flow(
            image,
            batch_size=1,
            save_to_dir=save_dir,
            save_prefix="aug_",
            save_format="jpeg",
        )

        # augmented_images_val = datagen.flow(
        #     image,
        #     batch_size=1,
        #     save_to_dir=validation_dir,
        #     save_prefix="aug_",
        #     save_format="jpeg",
        # )

        # Generate and save a number of augmented images
        for i in range(
            NUMBER_OF_VARIATIONS
        ):  # Specify the number of augmented images to generate
            next(
                augmented_images
            )  # Generates and saves the next batch of augmented images

        print(f"Augmented images saved to {save_dir}")
