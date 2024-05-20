"""Processess and prepares images for model training"""

import os
import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)
from config import IMAGE_SIZE, CLEANUP_DIRS, NUMBER_OF_AUGMENTATION_VARIANTS

directory_input = Path("src/product_images")
directory_output = Path("src/dataset")

product_classes = [
    filename for filename in os.listdir(directory_input) if filename != ".DS_Store"
]

print(product_classes)

# Clean up dirs
if CLEANUP_DIRS is True:
    if os.path.exists(Path("src/dataset/validation")):
        shutil.rmtree(Path("src/dataset/validation"))
    if os.path.exists(Path("src/dataset/train")):
        shutil.rmtree(Path("src/dataset/train"))

for fileclass in product_classes:
    # Loop through the files in the directory
    # Construct full file path
    filepath = os.path.join(directory_input, fileclass, "image.jpg")

    # Load your image
    image = Image.open(filepath)  # Make sure to provide the path to your image

    # Calculate new size, preserving aspect ratio
    aspect_ratio = image.width / image.height
    new_height = IMAGE_SIZE[1]
    new_width = int(new_height * aspect_ratio)
    if new_width > IMAGE_SIZE[0]:
        new_width = IMAGE_SIZE[0]
        new_height = int(new_width / aspect_ratio)

    image_resized = image.resize((new_width, new_height), Image.LANCZOS)

    # Convert the image to numpy array
    image_array = np.array(image_resized)

    # Ensure the array is four dimensional
    image_array = np.expand_dims(
        image_array, axis=0
    )  # This should be included to add the batch dimension

    # Copy original images to validation directory
    validation_dir = os.path.join(directory_output, "validation", fileclass)
    os.makedirs(validation_dir, exist_ok=True)

    # Save the resized image to the validation directory
    resized_image_path = os.path.join(validation_dir, "image.jpg")
    image_resized.save(resized_image_path)

    # Create augmentations from the original images into the train dataset
    for type in ["train"]:
        save_dir = Path(f"{directory_output}/{type}/{fileclass}/")
        os.makedirs(save_dir, exist_ok=True)

        # Create an ImageDataGenerator for data augmentation
        datagen = ImageDataGenerator(
            rotation_range=40,  # Degree range for random rotations
            width_shift_range=0.2,  # Range (as a fraction of total width) for horizontal shifts
            height_shift_range=0.2,  # Range (as a fraction of total height) for vertical shifts
            shear_range=0.2,  # Shearing intensity (shear angle in counter-clockwise direction)
            zoom_range=0.2,  # Range for random zoom
            # horizontal_flip=True,  # Randomly flip inputs horizontally
            fill_mode="constant",  # Strategy to fill newly created pixels
            cval=255,
        )

        # Generate batches of augmented images and save them to the specified directory
        augmented_images = datagen.flow(
            image_array,
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
            NUMBER_OF_AUGMENTATION_VARIANTS
        ):  # Specify the number of augmented images to generate
            next(
                augmented_images
            )  # Generates and saves the next batch of augmented images

        print(f"Augmented images saved to {save_dir}")
