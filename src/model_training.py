import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import numpy as np
import tensorflow as tf
from config import IMAGE_SIZE

DATASET_DIR = pathlib.Path("src/dataset")  #

BATCH_SIZE = 16

# labels are generated from the directory structure
train_dataset = image_dataset_from_directory(
    DATASET_DIR / "train", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)


validation_dataset = image_dataset_from_directory(
    DATASET_DIR / "validation", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)

test_dataset = image_dataset_from_directory(
    DATASET_DIR / "test_2", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE
)

N_CLASSES = len([name for name in os.listdir(pathlib.Path("src/dataset/validation"))])

# Define the model
## this is the model which wei et al used.
model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 2),
        ),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

# ------------------------------------------------------------------------
# Experiment 1: Different number of augmentations for training
# ------------------------------------------------------------------------

model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),  # Normalize the input images to [0, 1]
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)
# Results:
# 500 aug - Test accuracy: 0.778 Image size 180, 500 augmentations
# 100 aug - Test accuracy: 0.778 - loss: 2.6413 - Image size 180, 100 augmentations
# 80 aug - Test accurary:  0.667 - loss: 4.9318, 80 augmentations
# 50 aug - Test accuracy: 0.667 Image size 180, 50 augmentations
# 20 aug - accuracy: 0.6667 - loss: 1.7013 Image size 180, 20 augmentations
# 1 aug - Test accuracy: 0.333 Image size 180, 1 augmentations

# Results
# 1. Changing filter size from 64, 32 vs 32, 16 improves accuracy from accuracy: 0.7778 - loss: 1.0277 to  accuracy: 0.8889 - loss: 0.7551
# 2. Adding another max pooling layer, highlighting the most important features from the image
# reduces accuracy: accuracy: 0.6667 - loss: 2.0648
# 3. No max pooling layers: reduces accuracy: 0.6667 - loss: 5.9390
# 4. Kernel size from 3 to 5: reduces accuacy: accuracy: 0.6667 - loss: 1.1544
# 5. No padding: No difference: accuracy: 0.7778 - loss: 0.5554
# 6: Stride 2 on first conv2D layer: same accuracy
# 7: Removed second max pooling layer: Same
# 8: Increased pool_size from 2 to 3: accuracy: 0.8889 - loss: 0.5794
# 9: batch size does not affect accuracy


### FIRST WE TEST WEI MODEL ####

# Define the model
## this is the model which wei et al used.
model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)


# Test loss: 0.2908405065536499     # Validation loss: 0.0005306384991854429
# Test accuracy: 0.875              # Validation accuracy: 1.0


#### THEN WE TEST WITH ONE LESS MAXPOOLING LAYER

model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        # layers.MaxPooling2D(2, 2),  # MODEL WITHOUT THIS
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

# Test loss: 0.6465534567832947 # Validation loss: 0.0015060750301927328
# Test accuracy: 0.688          # Validation accuracy: 1.0

# This performs much worse


#### WHAT IF THE THIRD LAYER ALSO WAS DELETED ####


model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),
        # layers.Conv2D(filters=16, kernel_size=3, activation="relu"), #model without this also
        # layers.MaxPooling2D(2, 2),  # MODEL WITHOUT THIS
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

# Test loss: 0.9086869955062866     # Validation loss: 2.537705222493969e-05
# Test accuracy: 0.812              # Validation accuracy: 1.0


#### WHAT IF WE CHANGE TH FILTER NUMBERS ####


model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=32,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),
        # layers.Conv2D(filters=16, kernel_size=3, activation="relu"), #model without this also
        # layers.MaxPooling2D(2, 2),  # MODEL WITHOUT THIS
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)

# Test loss: 1.8487775325775146     # Validation loss: 0.0020548219326883554
# Test accuracy: 0.812              # Validation accuracy: 1.0


### WHAT IF WE INCLUDE ANOTHER LAYER + MAXPOOLING ####

model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)


# Test loss: 0.7882579565048218     # Validation loss: 0.221534863114357
# Test accuracy: 0.750              # Validation accuracy: 0.7777777910232544


### WHAT IF WE CHANGE KERNAL ####

model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=32, kernel_size=5, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=16, kernel_size=5, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)


# Test loss: 0.971632719039917    # Validation loss: 0.0019743291195482016
# Test accuracy: 0.812              # Validation accuracy: 0.1


#### SO THE FIRST MODEL WAS BEST!! IT IS THE BELOW CODE####

## this is the model which wei et al used.
model_3 = keras.Sequential(
    [
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        ),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=32, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Conv2D(filters=16, kernel_size=3, activation="relu"),
        layers.MaxPooling2D(2, 2),  # Saves the most important information
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(N_CLASSES, activation="softmax"),
    ]
)


# Test loss: 0.2908405065536499     # Validation loss: 0.0005306384991854429
# Test accuracy: 0.875              # Validation accuracy: 1.0


#### WHAT AUGMENTATION IS BEST
## with 100 ##
# Test loss: 0.2908405065536499     # Validation loss: 0.0005306384991854429
# Test accuracy: 0.875              # Validation accuracy: 1.0


## with 50 ##
# Test loss: 1.754159688949585      # Validation loss: 0.03422638028860092
# Test accuracy: 0.500              # Validation accuracy: 1.0


#### BELOW PRINTS THE RESULTS ###

# Compile model for multi-class with sparse categorical crossentropy
model_3.compile(
    optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
# Save the best model
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath=pathlib.Path("src/models/convnet_from_scratch.keras"),
        save_best_only=True,
        monitor="val_loss",
    )
]

# Fit the model
history = model_3.fit(
    train_dataset, epochs=5, validation_data=validation_dataset, callbacks=callbacks
)

validation_loss, validation_accuracy = model_3.evaluate(validation_dataset)
print(f"Validation loss: {validation_loss}")
print(f"Validation accuracy: {validation_accuracy}")

test_loss, test_acc = model_3.evaluate(test_dataset)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc:.3f}")
# weight_matrix = model.save_weights(filepath = 'src/model.weights.h5')
# print(weight_matrix)
