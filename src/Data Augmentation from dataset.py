# just like data augmentation but with different transformations
# again produce 8 images from 1
# keep some flips/rotations but also translate image, wrapping around + add noise
# apply translations randomly to avoid pattern recognition

# when you are appending a dataset with new data need to open old one, append and save as a new one
# can then delete the old one and rename new one to old name

import os
import pandas as pd
import tensorflow as tf
import numpy as np

num_images = 300  # need to edit based on data being loaded
num_transforms = 2  # number of different transformations including original
image_shape = (256, 256)


# create an empty tf dataset of the correct size or load dataset to append to
dataset = tf.data.Dataset.from_tensor_slices(([], []))
#dataset = tf.data.Dataset.load("run6 0 augmented 6.tfrecord")


# does the data augmentation
# randomly applied a transformation such as wraps and flips
def apply_random_transformations(image, max_shift=64):
    # ensure tensor of correct shape
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.ensure_shape(image, (256, 256, 1))

    # Randomly translates an image with wrap-around
    shift_x = tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32)
    shift_y = tf.random.uniform([], minval=-max_shift, maxval=max_shift, dtype=tf.int32)

    # apply random reflection
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Circular shift using tf.roll
    image = tf.roll(image, shift=[shift_y, shift_x], axis=[0, 1])

    # apply random rotation
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=k)

    return image


# function calls above function to create the number of augmented images that you want
def generate_augmented_images(image):
    augmented_images = [image]  # Keep original
    for _ in range(num_transforms-1):  # Generate 7 different augmentations to add
        augmented_images.append(apply_random_transformations(image))

    return augmented_images  # Returns a tensor of shape (8, H, W, C)


def load_tfrecord_to_numpy(tfrecord_path, dataset_size=350):  # need to edit dataset size as I increase it
    # Initialize empty arrays for images and labels
    images = np.zeros((dataset_size, 256, 256, 1), dtype=np.float32)  # Image shape (1200, 256, 256, 1)
    labels = np.zeros((dataset_size, 1), dtype=np.float32)  # Labels shape (1200, 1)

    # Load dataset using tf.data.Dataset.load() (This assumes your dataset is already serialized)
    dataset = tf.data.Dataset.load(tfrecord_path).shuffle(buffer_size=dataset_size)

    # Iterate over the dataset to collect images and labels
    for i, (image, label) in enumerate(dataset.take(dataset_size)):
        # Ensure the image has shape (256, 256, 3) for RGB
        if image.shape[-1] == 3:  # Check if it's RGB (3 channels)
            image = tf.image.rgb_to_grayscale(image)  # Convert to grayscale (1 channel)

        # Handle the case where the image might not have the expected shape (256, 256, 1)
        image = tf.image.resize(image, [256, 256])  # Resize to ensure it matches expected shape


        # Handle eager execution
        if isinstance(image, tf.Tensor):
            image = image.numpy()  # Convert image tensor to NumPy array
        if isinstance(label, tf.Tensor):
            label = label.numpy()  # Convert label tensor to NumPy array

        # Assign the numpy arrays to the corresponding index
        images[i] = image
        labels[i] = label

    return images, labels


# load tf dataset into numpy arrays
loaded_images, loaded_labels = load_tfrecord_to_numpy("Stackedx4 run6-8 300imgs.tfrecord", dataset_size=300)
print(loaded_labels.shape)
print(loaded_images.shape)


# load images and augment
for i in range(loaded_images.shape[0]):
    if i >= num_images:
        break

    # save images in list
    aug_images = generate_augmented_images(loaded_images[i])

    # now have augmented images saved in aug_images, need their label
    label = tf.cast(loaded_labels[i], tf.float32)
    labels = tf.repeat(label, num_transforms)
    # create a tf dataset with images
    new_data = tf.data.Dataset.from_tensor_slices((aug_images, labels))
    # Concatenate new data to the existing dataset
    dataset = dataset.concatenate(new_data)

tf.data.Dataset.save(dataset, "Stackedx4 Augmented run6-8 600imgs.tfrecord")
# now have a tfrecord saved to laptop



# checks - chatgpt
loaded_dataset = tf.data.Dataset.load("Stackedx4 Augmented run6-8 600imgs.tfrecord")

# Check the first few elements
for img, lbl in loaded_dataset.take(3):
    print("Image shape:", img.shape, "Label:", lbl.numpy())

original_count = sum(1 for _ in dataset)
loaded_count = sum(1 for _ in loaded_dataset)
print(f"Original dataset size: {original_count}, Loaded dataset size: {loaded_count}")
