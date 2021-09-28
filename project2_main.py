# -*- coding: utf-8 -*-
"""

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
tf.debugging.set_log_device_placement(True)


Created on Tue Jan 26 07:01:40 2021

@author: ZtY_Home

by

https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb#scrollTo=x9Kk1voUCaXJ

"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
# Helper libraries

logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# Import the Fashion MNIST dataset 70,000 grayscale images in 10 categories
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True,
                              with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# 60,000 images to train the network and 10,000 images to evaluate
class_names = metadata.features['label'].names
print("Class names: {}".format(class_names))
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# The value of each pixel in the image data is an integer in the range [0,255].
# For the model to work properly, these values need to be normalized
# to the range [0,1]


def normalize(images, labels):
    """Normalises a pixel value."""
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# Python’s map() is a built-in function that allows you to process and
# transform all the items in an iterable without using an explicit for loop,

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

"""
# plot an image to see what it looks like
# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28, 28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# Display the first 25 images from the training set
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(test_dataset.take(25)):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
"""

# Creating layers
linp = tf.keras.layers.Flatten(input_shape=(28, 28, 1))  # input
# the layer with 28*28=784 nodes
l1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)  # hidden (internal)
# ReLU type layer for nonlinear problems
l2 = tf.keras.layers.Dense(32, activation=tf.nn.relu)  # hidden (internal)
# ReLU type layer for nonlinear problems
lfin = tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # output
# softmax layer for providing probability

# Creating model
# Step 1 - adding layers to the model
model = tf.keras.Sequential([linp, l1, l2, lfin])
# Step 2 - setting parameters of the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Training the model
# Repeat forever by specifying dataset.repeat()
# the epochs parameter described below limits how long we perform training
# tells model.fit to use batches of 32
# dataset.shuffle(num_train_examples) randomizes the order
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(
    num_train_examples).batch(BATCH_SIZE)
# same for the test model
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# Actual training
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(
    num_train_examples/BATCH_SIZE))

# Accuracy evaluation
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# using the model - predicting name for all images
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

predictions.shape
predictions[0]
np.argmax(predictions[0])
test_labels[0]


# the full set of 10 class predictions
# function to draw the plots of images
def plot_image(i, predictions_array, true_labels, images):
    """Plotes the image."""
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# function plot the bar-chart of the names of images
def plot_value_array(i, predictions_array, true_label):
    """Plotes the probability bar chart."""
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# drawing 0th image:
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

# drawing 12th image:
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
shift = 15  # to arrange the shift in image display
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(shift+i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(shift+i, predictions, test_labels)

# Use the the trained model to make a prediction about a single img

# Grab an image from the test dataset
img = test_images[0]
print(img.shape)

# Add the image to a batch where it's the only member.
img = np.array([img])
print(img.shape)

# Predict the image
# model.predict returns a list of lists,
# one for each image in the batch of data
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
np.argmax(predictions_single[0])
