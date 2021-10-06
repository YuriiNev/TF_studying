"""project4."""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging  # creating log messages
import os  # os — to read files and directory structure
import matplotlib.pyplot as plt  # to plot the graph and display images in our training and validation data
import numpy as np
import math


def plotImages(images_arr):
    """Plot images that is what this function does.

    In the form of a grid with 1 row and 5 columns
    where images are placed in each column.
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Import TensorFlow Datasets
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

zip_dir = tf.keras.utils.get_file('cats_and_dogs_filterted.zip', origin=_URL,
                                  extract=True)

zip_dir_base = os.path.dirname(zip_dir)
print(zip_dir_base)

base_dir = os.path.join(os.path.dirname(zip_dir), 'cats_and_dogs_filtered')
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
# directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
# directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
# directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')


num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val

print("dog images: train, validation, altogether:",
      num_dogs_tr, num_dogs_val, num_dogs_tr+num_dogs_val)
print("cat images: train, validation, altogether:",
      num_cats_tr, num_cats_val, num_cats_tr+num_cats_val)
# Number of training examples to process before
# updating our models variables
BATCH_SIZE = 100
# Our training data consists of images with width
# of 150 pixels and height of 150 pixels
IMG_SHAPE = 150


""" MODEL AUGMENTATION """

""" Rotating images in the dataset
Example of how it could be applied is presented below"""
image_gen = ImageDataGenerator(rescale=1./255, rotation_range=45)
train_data_gen = image_gen.flow_from_directory(
  batch_size=BATCH_SIZE,
  directory=train_dir,
  shuffle=True,
  target_size=(IMG_SHAPE, IMG_SHAPE))
""" Printing the example """
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

""" Zooming images in the dataset
Example of how it could be applied is presented below"""
image_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.5)
train_data_gen = image_gen.flow_from_directory(
  batch_size=BATCH_SIZE,
  directory=train_dir,
  shuffle=True,
  target_size=(IMG_SHAPE, IMG_SHAPE))
""" Printing the example """
augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)

""" Augmenting the whole dataset"""

"""Images must be formatted into appropriately pre-processed floating
point tensors before being fed into the network. The steps involved in
preparing these images are:
1. Read images from the disk
2. Decode contents of these images and convert it into proper grid format
as per their RGB content
3. Convert them into floating point tensors
4. Rescale the tensors from values between 0 and 255 to values between
0 and 1, as neural networks prefer to deal with small input values.

Fortunately, all these tasks can be done using
the class tf.keras.preprocessing.image.ImageDataGenerator.
We can set this up in a couple of lines of code."""

image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
# Creating an object which will be applied to the dataset
# Next applying the object (method) to the dataset to obtain
# a new one with new prperties

train_data_gen = image_gen_train.flow_from_directory(
  batch_size=BATCH_SIZE,
  directory=train_dir,
  shuffle=True,
  target_size=(IMG_SHAPE, IMG_SHAPE),
  class_mode='binary')
# Returns a DirectoryIterator yielding tuples of (x, y)
# where x is a numpy array containing a batch of images with shape
# (batch_size, *target_size, channels) and y is a numpy array
# of corresponding labels.

"""Defining the validation dataset (without augmenting)"""

validation_image_generator = ImageDataGenerator(rescale=1./255)

"""After defining our generators for training and validation
images, flow_from_directory method will load images from the disk,
apply rescaling, and resize them using single line of code."""

val_data_gen = validation_image_generator.flow_from_directory(
  batch_size=BATCH_SIZE,
  directory=validation_dir,
  shuffle=True,
  target_size=(IMG_SHAPE, IMG_SHAPE),
  class_mode='binary')

""" Visualizing Training images """

augmented_images = [train_data_gen[0][0][0] for i in range(5)]
plotImages(augmented_images)



# [:5] = [0:5:1] 0 and :1 are defaults
# plotImages(sample_training_images[:5])  # Plot images 0-4


"""  Define the model """
"""  1. Define layers, make the list of layers
     2. Input the sequence (list) of layers into a model
     3. Compile the model
"""
# 1. Define layers
# in Conv2D: 32 - number of outputs, 3,3 - the size of the Kernel
# relu gives nonlinearity; input shape 150px x 150 px x 3 colors
layers = [tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                 input_shape=(150, 150, 3))]
layers.append(tf.keras.layers.MaxPooling2D(2, 2))

layers.append(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
layers.append(tf.keras.layers.MaxPooling2D(2, 2))

layers.append(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
layers.append(tf.keras.layers.MaxPooling2D(2, 2))

layers.append(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
layers.append(tf.keras.layers.MaxPooling2D(2, 2))

layers.append(tf.keras.layers.Dropout(0.25))  # turns off 25% of nodes randomly

layers.append(tf.keras.layers.Flatten())

layers.append(tf.keras.layers.Dropout(0.25))
layers.append(tf.keras.layers.Dense(512, activation='relu'))
layers.append(tf.keras.layers.Dense(2))  # default activation is linear
layers.append(tf.keras.layers.Dense(2, activation='softmax'))

# 2. Input the sequence (list) of layers into a model
model = tf.keras.models.Sequential(layers)

# 3. Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

""" Model Summary """
model.summary()

EPOCHS = 85
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)
# np.ceil(number) is the smallest integer which is more than the number
# using fit_generator instead of fit for the data from generator
# if I get it right it is possible to use model.fit, but less productive

""" Visualizing results of the training """

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()







