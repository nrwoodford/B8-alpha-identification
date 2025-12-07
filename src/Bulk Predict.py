import keras
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import re

"""
Predicts on entire runs
Save data input in one folder, with subfolders for each angle
Inputs are the folder path and names for the output files
Outputs the count and variance for each angle
For high angle it will also output the file names which is claims to have alphas in a csv
High Angle threshold set at 46 based on testing, batch size can be increased if lots of data
"""

# Variables to set:
folder_path = "Data/Will Run7"  # Input name of folder with subfolders of images at each angle
output_predictions_file = "Run7 Predictions.csv"  # Filename.csv to save outputs to
output_alpha_images = "Run7 Alphas test.csv"  # Filename.csv to save paths of high angle images with alphas for checking

high_angle_threshold = 46  # angle at which runs designated high angle
high_angle_model = "Finetuned Model - High Angle Optimised"  # model for high angle runs
low_angle_model = "Finetuned Model - Low Angle Optimised"  # model for low angle runs

image_shape = (256, 256)
batch_size = 128  # Number of images to be processed at a time by the model


# function to turn images into all 0s and 1s to remove issues with brightness
# input and output both images, sets any value in image greater than threshold to 1, any below to 0
def binary_image(image, threshold=0.05):
    binary = tf.cast(image > threshold, tf.float32)
    return binary


# preprocessing code to open image path, convert to array and normalise
# Outputs an image as np.array
def image_preprocessing(image_path):
    image = Image.open(image_path)  # Convert to grayscale
    image = image.resize(image_shape)
    image = np.array(image)
    image = image/255  # normalise
    image = binary_image(image, 0.05)  # set every value to 0 or 1
    image = np.expand_dims(image, axis=-1)
    return image


folder_results = []
high_angle_ones = []

# Loop through subfolders - processing each individually
for subfolder in sorted(os.listdir(folder_path)):
    subfolder_path = os.path.join(folder_path, subfolder)

    if not os.path.isdir(subfolder_path):
        continue  # Check this is a folder

    match = re.search(r"(\d+)", subfolder)  # Extract degrees the run was performed at
    angle = int(match.group(1)) if match else None  # Convert to int only if there's a match

    # Different models for high and low angles - defaults to low angle
    if angle >= high_angle_threshold:
        model = keras.models.load_model(high_angle_model)
        model.trainable = False
        high_angle = True
        print(f"Processing high angle folder: {subfolder}")
    else:
        model = keras.models.load_model(low_angle_model)
        model.trainable = False
        high_angle = False
        print(f"Processing low angle folder: {subfolder}")

    # Collect images in batches
    img_arrays = []
    predictions_list = []
    img_paths = []

    # Now in a subfolder, process each image in it
    for img_name in sorted(os.listdir(subfolder_path)):
        img_path = os.path.join(subfolder_path, img_name)

        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue  # Skip non-image files

        img_arrays.append(image_preprocessing(img_path))
        img_paths.append(img_path)

    # Take images and input to model for predictions
    batch_array = np.array(img_arrays)
    print(batch_array.shape)
    batch_predictions = model.predict(batch_array, batch_size=batch_size)
    batch_predictions = np.round(batch_predictions)  # Round to nearest integer
    # Store all predictions
    predictions_list.extend([max(0, int(item))] for item in batch_predictions)

    #  If a high angle run, save the names of files it claims to have alphas to be checked by hand
    if high_angle:
        # Save image paths where prediction > 0
        for path, prediction in zip(img_paths, batch_predictions):
            if prediction > 0:
                high_angle_ones.append([subfolder, path, prediction])
                # For high angle runs, save names of images with alphas to be checked

    # Compute total prediction sum and variance for the subfolder
    print(predictions_list)
    total_prediction = np.sum(predictions_list)
    variance_prediction = np.var(predictions_list)

    # Save the results per folder
    folder_results.append([angle, total_prediction, variance_prediction])


# Write the results from each folder into a dataframe, then a csv
df_folders = pd.DataFrame(folder_results, columns=["Degree", "Sum", "Variance"])
df_folders.to_csv(output_predictions_file, index=False)

# Save alpha image paths
if high_angle_ones:
    df_alpha_images = pd.DataFrame(high_angle_ones, columns=["Folder", "Image Path", "Prediction"])
    df_alpha_images.to_csv(output_alpha_images, index=False)

print("Outputs saved to: " + output_predictions_file)

# Plot prediction sum against angle
# Load the CSV file with folder predictions
df = pd.read_csv(output_predictions_file)

# Ensure sorted by angle
df = df.sort_values(by="Degree")

# Plot total prediction sum vs. angle
plt.figure(figsize=(8, 5))
plt.plot(df["Degree"], df["Sum"], marker="o", linestyle="-", color="b", label="Total Prediction")

plt.xlabel("Angle (degrees)")
plt.ylabel("Total Alphas detected")
plt.title("Alphas detected vs. Angle")
plt.legend()

plt.grid(True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.show()
