import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random

"""
Generates a synthetic dataset by adding alphas to background images
"""


# Generates an image with a number of alpha particles
def generate_alpha_particle_image(size=256, num_particles=0, intensity=1, spread=5):

    image = np.zeros((size, size), dtype=np.float32)  # Black background

    for _ in range(num_particles):
        x, y = np.random.randint(0, size, size=2)  # Random center position
        intensity = np.random.uniform(0.75, 1)  # make intensity high but not always 1

        for i in range(-spread * 2, spread * 2):
            for j in range(-spread * 2, spread * 2):
                xi, yj = x + i, y + j
                if 0 <= xi < size and 0 <= yj < size:
                    distance = np.exp(-((i ** 2 + j ** 2) / (2 * spread ** 2)))
                    image[yj, xi] += intensity * distance  # Apply Gaussian shape

    image = np.clip(image, 0, 1)  # Ensure values are within [0,1]

    return image


# Generates an image with a number of alphas added randomly to a sample of background
# load a random real image with no alphas to use as the background then add in a random number of alphas
def generate_noisy_image(size=256, num_particles=0, intensity=1, spread=1):

    folder_path = "Data/Background"
    image_files = [f for f in os.listdir(folder_path)]
    random_image = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image)
    image = Image.open(image_path)
    # open image and use as background to add alpha over
    image = np.array(image)/255  # normalise and convert to np array

    # repeat so double background
    random_image2 = random.choice(image_files)
    image_path = os.path.join(folder_path, random_image2)
    image2 = Image.open(image_path)
    # open image and use as background to add alpha over
    image2 = np.array(image2) / 255  # normalise and convert to np array

    image = image + image2

    for _ in range(num_particles):
        x, y = np.random.randint(0, size, size=2)  # Random center position
        intensity = np.random.uniform(0.75, 1)
        # make intensity high but not always 1

        for i in range(-spread * 2, spread * 2):
            for j in range(-spread * 2, spread * 2):
                xi, yj = x + i, y + j
                if 0 <= xi < size and 0 <= yj < size:
                    distance = np.exp(-((i ** 2 + j ** 2) / (2 * spread ** 2)))
                    image[yj, xi] += intensity * distance  # Apply Gaussian shape

    image = np.clip(image, 0, 1)  # Ensure values are within [0,1]
    return image

# computer cant deal with creating 10,000 image dataset all at once
# instead create 10 datasets of 1000, save one at a time, then load one at a time and concatenate into big one

num_samples = 2000
num_slices = 10
num = 1000

dataset = tf.data.Dataset.from_tensor_slices(([], []))

#for j in range(num_slices):
    #dataset = tf.data.Dataset.from_tensor_slices(([], [])).batch(32)

    #if j == num_slices-1:
for i in range(num_samples):
    num_particles = np.random.randint(0, 20)  # Random count 0-19
    img = generate_alpha_particle_image(num_particles=num_particles)  # no noise for 10%
    img = img.reshape(256, 256, 1)
    image = tf.cast(img, tf.float32)

    num_particles = tf.cast(num_particles, tf.float32)

    new_data = tf.data.Dataset.from_tensors((image, num_particles))
    dataset = dataset.concatenate(new_data)
    #else:
        #for i in range(num):
            #num_particles = np.random.randint(0, 20)  # Random count 0-19
            #img = generate_noisy_image(num_particles=num_particles)
            #img = img.reshape(256, 256, 1)
            #image = tf.cast(img, tf.float32)

            #num_particles = tf.cast(num_particles, tf.float32)

            #new_data = tf.data.Dataset.from_tensors((image, num_particles))
            #dataset = dataset.concatenate(new_data)

#tf.data.Dataset.save(dataset, "large_noisy_synthetic_data" + str(j) + ".tfrecord")


#dataset = tf.data.Dataset.from_tensor_slices(([], [])).batch(32)
#for i in range(num_slices):
#    temp = tf.data.Dataset.load("large_noisy_synthetic_data" + str(i) + ".tfrecord")
#    dataset = dataset.concatenate(temp)


tf.data.Dataset.save(dataset, "stacked_synthetic_data.tfrecord")
