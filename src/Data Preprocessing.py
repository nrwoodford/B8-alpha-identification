# very similar to data augmentation but just use original files
# want to check the augmentation is actually helping
import os
import pandas as pd
import numpy as np
import tensorflow as tf

num_images = 300  # need to edit based on data being loaded
image_shape = (256, 256)
image_dir = ("Data for Stacking/Run10 160")

# create an empty tf dataset of the correct size or load dataset to append to
#dataset = tf.data.Dataset.from_tensor_slices(([], []))
dataset = tf.data.Dataset.load("Stackedx2 run6-9 900imgs.tfrecord")

# read labels into csv
df = pd.read_csv("Data for Stacking/Run10 160 labels.csv", header=None)
Labels = df.to_numpy().reshape(num_images, 1).astype('int')  # edit to size of data being loaded


n = 0
temp_image = np.zeros((256, 256, 1))
temp_label = 0

# load images and augment
for i, filename in enumerate(os.listdir(image_dir)):

    if i >= num_images:
        break

    filepath = os.path.join(image_dir, filename)
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=1)  # chatGPT code
    image = tf.cast(image, tf.float32)
    image = np.array(image)

    label = Labels[i]
    # this is for when i dont want to stack images

    label = tf.cast(Labels[i], tf.float32)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.ensure_shape(image, (256, 256, 1))
    new_data = tf.data.Dataset.from_tensors((image, label))
    # Concatenate new data to the existing dataset
    dataset = dataset.concatenate(new_data)
    """

    # this is for when i want to stack images
    if n == 1:
        # create a tf dataset with images
        temp_label += label
        temp_image += image
        # convert everything to tensors to be saved
        temp_label = tf.cast(temp_label, tf.float32)
        temp_image = tf.convert_to_tensor(temp_image, dtype=tf.float32)

        temp_image = tf.ensure_shape(temp_image, (256, 256, 1))

        new_data = tf.data.Dataset.from_tensors((temp_image, temp_label))
        # Concatenate new data to the existing dataset
        dataset = dataset.concatenate(new_data)

        n = 0
        temp_image = np.zeros((256, 256, 1))
        temp_label = 0  # reset everything to 0
    else:
        temp_image += image
        temp_label += label
        n += 1  # save to add to next group
    """

tf.data.Dataset.save(dataset, "Stackedx2 run6-10 1200imgs.tfrecord")
# now have a tfrecord saved to laptop

# checks - chatgpt
loaded_dataset = tf.data.Dataset.load("Stackedx2 run6-10 1200imgs.tfrecord")

# Check the first few elements
for img, lbl in loaded_dataset.take(3):
    print("Image shape:", img.shape, "Label:", lbl.numpy())

original_count = sum(1 for _ in dataset)
loaded_count = sum(1 for _ in loaded_dataset)
print(f"Original dataset size: {original_count}, Loaded dataset size: {loaded_count}")
