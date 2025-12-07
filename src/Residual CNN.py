import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np
import os
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Add, Input
from tensorflow.keras.callbacks import LearningRateScheduler

"""
Trains base model on synthetic data and model with best validation mean absolute error
Defines model architecture and then trains it on a dataset which has been loaded into images and labels
Images are preprocessed so all values are 0 or 1
"""


# Takes in an image and make all pixels either 0 or 1
def binary_image(image, threshold=0.05):
    binary = tf.cast(image > threshold, tf.float32)
    # image > threshold gives a Boolean value at each pixel
    # Casting it as a tf.float32 makes True=1 and False=0
    return binary


# Load tfrecord dataset into np.arrays - written with help of Chat-GPT
# Takes in path to the dataset file, and the size of the dataset
# Outputs arrays images (shape: dataset_size, 256, 256, 1) and labels (shape: dataset_size, 1)
def load_tfrecord_to_numpy(tfrecord_path, dataset_size):
    # Initialize empty arrays for images and labels
    images = np.zeros((dataset_size, 256, 256, 1), dtype=np.float32)
    labels = np.zeros((dataset_size, 1), dtype=np.float32)

    # Load dataset and shuffle
    dataset = tf.data.Dataset.load(tfrecord_path).shuffle(buffer_size=dataset_size)

    # Iterate over the dataset to collect images and labels
    for i, (image, label) in enumerate(dataset.take(dataset_size)):

        if image.shape[-1] == 3:  # Check if it's RGB (3 channels)
            image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale (1 channel)

        image = tf.image.resize(image, [256, 256])  # Ensure matches expected image shape
        image = tf.cast(image, tf.float32)
        image = image/255  # Normalise values to between 0 and 1
        image = binary_image(image, 0.05)  # Process all values to 0 or 1

        # Ensure output is an array
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        if isinstance(label, tf.Tensor):
            label = label.numpy()

        images[i] = image
        labels[i] = label

    return images, labels


# Load data into numpy arrays
loaded_images, loaded_labels = load_tfrecord_to_numpy("Datasets/improved_large_synthetic_data.tfrecord",
                                                      dataset_size=10000)
print(loaded_labels[0])
print(loaded_images.shape)


# the model:

# Input layer
inputs = Input(shape=(256, 256, 1))

# First conv + pooling
x = Conv2D(32, (5, 5), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Residual block with two convolutional layers
residual = x
x = Conv2D(32, (5, 5), padding='same', activation='relu')(x)
x = Conv2D(32, (5, 5), padding='same')(x)
x = Add()([x, residual])  # Skip connection
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Flatten and Dense layers
x = Flatten()(x)
x = Dense(64)(x)
x = Dropout(0.15)(x)  # Dropout to reduce overfitting
outputs = Dense(1)(x)  # Output of shape (1)

# Compile model with Adam optimiser and mean absolute error as loss function
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss="mae", optimizer='adam', metrics=['mae'])

model.summary()


# Define function to decay learning rate
def exponential_decay(epoch, lr):
    if epoch < 5:  # start big as mae starts very large
        return 0.005
    if epoch > 35:  # half lr again for final epochs
        return 0.00005
    if epoch > 20:  # eventually settle at 0.0001
        return 0.0001
    if epoch >= 12:  # after 12 epochs start reducing exponentially
        return lr * tf.math.exp(-0.05 * (epoch-11))
    return 0.001


# Create the scheduler
lr_scheduler = LearningRateScheduler(exponential_decay)


# Define model checkpoint to save the model with the best validation MAE
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    "Base Residual Model",  # Filename
    monitor="val_mae",
    save_best_only=True,
    mode="min"  # Lower MAE is better
)


# Train the model for 50 epochs, with 20% of the data kept aside for validation
history = model.fit(loaded_images, loaded_labels, validation_split=0.2, batch_size=32, epochs=50,
                    callbacks=[lr_scheduler, checkpoint_best])


loss = history.history['mae']
val_loss = history.history['val_mae']

print(loss)
print(val_loss)

epochs = range(1, len(loss) + 1)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Plot training and validation loss on same graph
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label="Training MAE")
plt.plot(epochs, val_loss, 'ro-', label="Validation MAE")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Means Absolute Error")
plt.legend()

plt.show()
