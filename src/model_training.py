import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import tensorflow as tf

new_base_dir = pathlib.Path("src/dataset")


# labels are generated from the directory structure
train_dataset = image_dataset_from_directory(
    new_base_dir / "train", image_size=(180, 180), batch_size=16
)

test_dataset = image_dataset_from_directory(
    new_base_dir / "validation", image_size=(180, 180), batch_size=16
)

N_CLASSES = len([name for name in os.listdir("src/dataset/validation")])

# Define the model
# TODO: understand this better
model = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),  # Normalize the input images to [0, 1]
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(180, 180, 3)),
        # layers.MaxPooling2D(2, 2),
        # layers.Conv2D(64, (3, 3), activation="relu"),
        # layers.MaxPooling2D(2, 2),
        # layers.Conv2D(128, (3, 3), activation="relu"),
        # layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),  #
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

# Compile model for multi-class with sparse categorical crossentropy
model.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
# Save the best model
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="src/convnet_from_scratch.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]

# Fit the model
history = model.fit(
    train_dataset, epochs=5, validation_data=test_dataset, callbacks=callbacks
)

test_loss, test_accuracy = model.evaluate(test_dataset)

print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
