# just like data augmentation but with different transformations
# again produce 8 images from 1
# keep some flips/rotations but also translate image, wrapping around + add noise
# apply translations randomly to avoid pattern recognition

# when you are appending a dataset with new data need to open old one, append and save as a new one
# can then delete the old one and rename new one to old name

import os
import pandas as pd
import tensorflow as tf

"""
Takes a dataset and augments each image a certain number of times
Creates a new dataset of the old images and new augmented images
"""

num_images = 100  # need to edit based on data being loaded
num_transforms = 8  # number of different transformations including original
image_shape = (256, 256)
image_dir = "Data/Will Run6/tao05_0deg"

# create an empty tf dataset of the correct size or load dataset to append to
dataset = tf.data.Dataset.from_tensor_slices(([], []))
#dataset = tf.data.Dataset.load("run6 0 augmented 6.tfrecord")

# read labels into csv
df = pd.read_csv("Data/Will Run6/ta05_0deg labels.csv", header=None)
Labels = df.to_numpy().reshape(100, 1).astype('int')  # edit to size of data being loaded


# Does the data augmentation - takes an image as an input
# Randomly applies a transformation such as wraps and flips
# Outputs a new image which has been augmented
def apply_random_transformations(image, max_shift=64):

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


# Generates a certain number of augmented images from an input image and outputs them in a tensor
def generate_augmented_images(image):
    augmented_images = [image]  # Keep original
    for _ in range(num_transforms-1):  # Generate 7 different augmentations to add
        augmented_images.append(apply_random_transformations(image))

    return augmented_images  # Returns a tensor of shape (num_transforms, 256, 256, 1)


# Load images in the folder and augments them to create new dataset
for i, filename in enumerate(os.listdir(image_dir)):
    if i >= num_images:
        break
    filepath = os.path.join(image_dir, filename)
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image)
    image = tf.cast(image, tf.float32)  # Reads image into a tensor

    # Save images in list of tensors
    aug_images = generate_augmented_images(image)

    # now have augmented images saved in aug_images, need their label
    label = tf.cast(Labels[i], tf.float32)
    labels = tf.repeat(label, num_transforms)  # Repeat old label as many times as needed
    # create a tf dataset with images
    new_data = tf.data.Dataset.from_tensor_slices((aug_images, labels))
    # Concatenate new data to the existing dataset
    dataset = dataset.concatenate(new_data)

tf.data.Dataset.save(dataset, "run6 0 augmented 8.tfrecord")


# Dataset checks - chatgpt
loaded_dataset = tf.data.Dataset.load("run6 0 augmented 8.tfrecord")

# Check the first few elements
for img, lbl in loaded_dataset.take(3):
    print("Image shape:", img.shape, "Label:", lbl.numpy())

# Check the dataset sizes match
original_count = sum(1 for _ in dataset)
loaded_count = sum(1 for _ in loaded_dataset)
print(f"Original dataset size: {original_count}, Loaded dataset size: {loaded_count}")
