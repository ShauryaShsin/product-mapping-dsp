import os
import numpy as np
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    img_to_array,
    load_img,
)

NUMBER_OF_VARIATIONS = 10

directory = "dataset/kims_halv_salt"

# Loop through the files in the directory
for filename in os.listdir(directory):
    # Construct full file path
    filepath = os.path.join(directory, filename)
    # Load your image
    image = load_img(filepath)  # Make sure to provide the path to your image
    image = img_to_array(image)  # Convert the image to numpy array
    image = np.expand_dims(
        image, axis=0
    )  # Add a new axis to make the image array four dimensional

    # Define the directory to save augmented images
    save_dir = "dataset/train_data/sour_cream"
    os.makedirs(save_dir, exist_ok=True)

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,  # Degree range for random rotations
        width_shift_range=0.2,  # Range (as a fraction of total width) for horizontal shifts
        height_shift_range=0.2,  # Range (as a fraction of total height) for vertical shifts
        shear_range=0.2,  # Shearing intensity (shear angle in counter-clockwise direction)
        zoom_range=0.2,  # Range for random zoom
        horizontal_flip=True,  # Randomly flip inputs horizontally
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

    # Generate and save a number of augmented images
    for i in range(
        NUMBER_OF_VARIATIONS
    ):  # Specify the number of augmented images to generate
        next(augmented_images)  # Generates and saves the next batch of augmented images

    print(f"Augmented images saved to {save_dir}")
