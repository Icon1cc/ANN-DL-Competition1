import numpy as np
from keras.preprocessing import image
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import os
from PIL import Image


def augment_image(X, rotation = 90, zoom = 0.2,  flip = "horizontal", translation = 0.2):
        # Returns randomly rotated, zoomed and flipped image(s) 
    # X: image(s) to be augmented
    # rotation: rotation range in degrees
    # zoom: zoom range
    # flip: "horizontal", "vertical" or None

    rotation_ = tf.keras.Sequential([
    tfkl.RandomRotation(rotation),
    ])

    zoom_ = tf.keras.Sequential([
    tfkl.RandomZoom(zoom),
    ])


    flip_ = tf.keras.Sequential([
    tfkl.RandomFlip(flip),
    ])

    return rotation_((flip_(zoom_(X))))


def equalize_classdistribution(X_healthy, X_unhealthy, y_healthy, y_unhealthy):
    # Equalize the class distribution by augmenting and increasing the smaller class
    # X_healthy: healthy images
    # X_unhealthy: unhealthy images
    # y_healthy: healthy labels
    # y_unhealthy: unhealthy labels
    #                                                               
    # Returns the augmented healthy and unhealthy images and labels tensors of int type in range [0, 255]
    #                                                               


    # Augment the smaller class
    diff = len(X_healthy) - len(X_unhealthy)
    if diff > 0:
        X = X_unhealthy.copy()
        y = y_unhealthy.copy()
        random_indices = np.random.randint(0, len(X), size=diff)
        X_unhealthy_augmented = augment_image(X[random_indices], rotation = 90, zoom = 0.2,  flip = "horizontal")
        X_unhealthy_augmented = X_unhealthy_augmented.numpy()
        y_unhealthy_augmented = y[random_indices]
        # Concatenate the augmented diseased plants with the diseased plants
        X_unhealthy = np.concatenate((X_unhealthy, X_unhealthy_augmented.astype(int)))
        y_unhealthy = np.concatenate((y_unhealthy, y_unhealthy_augmented))

    else:
        X = X_healthy.copy()
        y = y_healthy.copy()
        diff = -diff
        random_indices = np.random.randint(0, len(X), size=diff)
        X_healthy_augmented = augment_image(X[random_indices], rotation = 90, zoom = 0.2,  flip = "horizontal")
        X_healthy_augmented = X_healthy_augmented.numpy()
        y_healthy_augmented = y[random_indices]
        # Concatenate the augmented diseased plants with the diseased plants
        X_healthy = np.concatenate((X_healthy, X_healthy_augmented.astype(int)))
        y_healthy = np.concatenate((y_healthy, y_healthy_augmented))

    return X_healthy.astype(int), X_unhealthy.astype(int), y_healthy, y_unhealthy



def load_images_from_folder(folder_path, target_size=(96, 96)):
    images = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            img = img.resize(target_size)
            img_array = image.img_to_array(img)
            images.append(img_array)

    return np.stack(images, axis=0)

healthy_path = os.getcwd() + "/healthy"
unhealthy_path = os.getcwd() + "/unhealthy"
X_healthy = load_images_from_folder(healthy_path)
X_unhealthy = load_images_from_folder(unhealthy_path)

print(f"Number of healthy images: {len(X_healthy)}")
print(f"Number of unhealthy images: {len(X_unhealthy)}")

x = np.concatenate((X_healthy, X_unhealthy))
y = np.concatenate((np.zeros(len(X_healthy)), np.ones(len(X_unhealthy))))
np.savez_compressed("clean.npz", data=x, labels=y)
print(X_healthy.shape)
print(X_unhealthy.shape)

X_healthy, X_unhealthy, y_healthy, y_unhealthy = equalize_classdistribution(X_healthy, X_unhealthy, np.zeros(len(X_healthy)), np.ones(len(X_unhealthy)))

X = np.concatenate((X_healthy, X_unhealthy))
y = np.concatenate((y_healthy, y_unhealthy))

#Plot the last 10 images in X
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 5)

for i in range(2):
    for j in range(5):
        ax[i, j].imshow(X[-(i*5+j+1)])
        ax[i, j].set_title(f"Label: {y[-(i*5+j+1)]}")
        ax[i, j].axis("off")

plt.show()


# Save the dataset as npz file
np.savez_compressed("clean_equaldist.npz", data=X, labels=y)