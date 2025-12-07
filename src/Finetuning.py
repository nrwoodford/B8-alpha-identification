import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Loads pretrained model and finetunes it on real data

# Processing functions as in CNN Training
def binary_image(image, threshold):
    binary = tf.cast(image > threshold, tf.float32)
    return binary


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
        #image = preprocessing(image)  # need to normalise real data
        # if i remove this i need to divide by 255!!
        image = image/255  # normalise
        image = binary_image(image, 0.05)  # normalise to all 0s and 1s

        # Handle eager execution
        if isinstance(image, tf.Tensor):
            image = image.numpy()  # Convert image tensor to NumPy array
        if isinstance(label, tf.Tensor):
            label = label.numpy()  # Convert label tensor to NumPy array

        # Assign the numpy arrays to the corresponding index
        images[i] = image
        labels[i] = label

    return images, labels


# Load data into numpy arrays
loaded_images, loaded_labels = load_tfrecord_to_numpy("Datasets/Augmented High Angle Training Set.tfrecord"
                                                      , dataset_size=2124)
print(loaded_labels[0])
print(loaded_images.shape)


# schedule learning rate to reduce after a plateau in learning
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',   # Metric to watch
    factor=0.5,           # Reduce LR by this factor (e.g., 0.5 = half)
    patience=3,           # Number of epochs with no improvement before reducing LR
    min_lr=1e-7,          # Lower bound for LR
    verbose=1             # Print when LR is reduced
)

# Create the scheduler
lr_scheduler = reduce_lr

# Define model checkpoint to save the model with the best validation MAE
checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
    "Finetuned High Angle Model",  # Filename for best model
    monitor="val_mae",  # Monitor validation MAE
    save_best_only=True,  # Save only the best model
    mode="min"  # Lower MAE is better
)


model = keras.models.load_model("Base Residual Model")  # Load model to be finetuned

# Train Model
history = model.fit(loaded_images, loaded_labels, validation_split=0.2, batch_size=32, epochs=40,
                    callbacks=[lr_scheduler, checkpoint_best])


loss = history.history['mae']
val_loss = history.history['val_mae']

print(loss)
print(val_loss)

epochs = range(1, len(loss) + 1)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Plot Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'bo-', label="Training MAE")  # 'bo-' means blue dots and line
plt.plot(epochs, val_loss, 'r-', label="Validation MAE")  # 'r-' means red line
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Means Absolute Error")
plt.legend()

plt.show()

[3.1049437522888184, 1.5235817432403564, 1.3020470142364502, 1.0442684888839722, 0.7367590665817261, 0.43836477398872375, 0.3452761769294739, 0.30159544944763184, 0.3024292588233948, 0.2467622011899948, 0.2171040028333664, 0.22282476723194122, 0.20711325109004974, 0.2040838897228241, 0.18882864713668823, 0.19123515486717224, 0.1819644272327423, 0.1864320933818817, 0.20701012015342712, 0.1992725282907486, 0.16576552391052246, 0.15867386758327484, 0.13944007456302643, 0.14164778590202332, 0.14673873782157898, 0.13502472639083862, 0.13244564831256866, 0.13492624461650848, 0.13083279132843018, 0.1419590413570404]
[1.4431592226028442, 1.3181586265563965, 1.191505789756775, 1.0492918491363525, 0.7780588865280151, 0.6160332560539246, 0.5993934273719788, 0.6674306392669678, 0.5936957001686096, 0.5909548401832581, 0.5829469561576843, 0.5614420771598816, 0.5804851055145264, 0.5631015300750732, 0.5598049163818359, 0.5539981126785278, 0.538719117641449, 0.5455434918403625, 0.5601051449775696, 0.5442337989807129, 0.540777862071991, 0.5368677973747253, 0.5305777788162231, 0.5305959582328796, 0.527756929397583, 0.5356347560882568, 0.5259779095649719, 0.5255923867225647, 0.5327533483505249, 0.5241966843605042]
