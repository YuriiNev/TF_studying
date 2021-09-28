"""project3."""
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf

# Import TensorFlow Datasets
import tensorflow_datasets as tfds

tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# loading daata
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True,
                              with_info=True)
# naming data, thoough I wonder why?
train_dataset = dataset['train']
test_dataset = dataset['test']

"""
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']
"""
class_names = metadata.features['label'].names
# 60,000 images to train the network and 10,000 images to evaluate
print("Class names: {}".format(class_names))

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))

# The value of each pixel in the image data is an integer in the range [0,255].
# For the model to work properly, these values need to be normalized
# to the range [0,1]


def normalize(images, labels):
    """Normalises a pixel value from 0-255 to 0-1."""
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# The first time you use the dataset, the images will be loaded from disk
# Caching will keep them in memory, making training faster
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()

# Take first 30 images with test_dataset.take(30)
# A crude way of printing imeages 5-30
plt.figure(figsize=(10, 10))
i = 0
initial = 5  # initial image to plot
for (image, label) in test_dataset.take(30):
    if i >= initial:
        image = image.numpy().reshape((28, 28))
        plt.subplot(5, 5, i-initial+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
    i += 1
plt.show()

# General info about lists in Python
"""
L[a], L[b] = L[b], L[a] if shifting is needed
more information about shifting objects in lists via the link:
https://coderoad.ru/39167057/%D0%9B%D1%83%D1%87%D1%88%D0%B8%D0
%B9-%D1%81%D0%BF%D0%BE%D1%81%D0%BE%D0%B1-%D0%BF%D0%BE%D0%BC%D0
%B5%D0%BD%D1%8F%D1%82%D1%8C-%D0%BC%D0%B5%D1%81%D1%82%D0%B0%D0%
BC%D0%B8-%D1%8D%D0%BB%D0%B5%D0%BC%D0%B5%D0%BD%D1%82%D1%8B-%D0%
B2-%D1%81%D0%BF%D0%B8%D1%81%D0%BA%D0%B5 """
# Creatung a model-layers part
""" Creating the list of layers, starting with layer 0.
THE ORDER DOES METTER!!!
Adding new layers in the end of existing list """

layers = [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                 padding='same', activation=tf.nn.relu,
                                 input_shape=(28, 28, 1))]
# this layer creates 32 convoluted images 28x28 pixels: 1 image for each filter
# (overall 28*28*32=25088 nodes)
# obtained by applying 3x3 filter --> the values in the filter are changing during training
# so to minimize the loss function
# more info: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           strides=2))
# MaxPoolimng = reducing the size of the image; applying 2x2 frame
# Taking the maximum value from the frame and put it into the new image
# shifting the frame by 2 (strides = n means strides = (n, n));
# overall  28/2*28/2*32=25088/4=6272 nodes in 32 batches

layers.append(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                     activation=tf.nn.relu))
# the same as 1st, but now it is 14*14*64=12544 nodes in 64 batches

layers.append(tf.keras.layers.MaxPooling2D((2, 2), strides=2))
# the same as 2nd, 14/2*14/2*64=3136 nodes; 7*7 nodes in 64 batches

layers.append(tf.keras.layers.Flatten())
# flattens the last layer

layers.append(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# add the layer with 128 nodes operating by relu rule
layers.append(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# the list of layers above is the argument in Sequential
model = tf.keras.Sequential(layers)

# Compilation of the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

BATCH_SIZE = 32
# .batch(BATCH_SIZE) tells model.fit to use batches of BATCH_SIZE
# images and labels when updating the model variables
# .shuffle(num_train_examples) randomizes the order of train_examples
# .repeat() makes it to repeat forever, until the number of epochs is reached
# .cash uploads dataset into cache
# the order of .functions applied to the dataset object seams
# does not matter, except should end by cache().repeat() or .cache()
train_dataset = train_dataset.shuffle(num_train_examples).batch(
                BATCH_SIZE).cache().repeat()
test_dataset = test_dataset.batch(BATCH_SIZE).cache()

# TRAINING THE MODEL
# epoch - the numder of uses of the whole dataset
# steps_per_epoch - the number of changes of batches (must be integer)
model.fit(train_dataset, epochs=10,
          steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# EVALUATION OF THE MODEL
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.
                                          ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

number_of_batch = 1  # the number of batch we take, starting with 1
for test_images, test_labels in test_dataset.take(number_of_batch):
    test_images = test_images.numpy()  # .numpy() converts a Tensor
    test_labels = test_labels.numpy()  # to a Numpy array, so images and
    predictions = model.predict(x=test_images, batch_size=BATCH_SIZE)
# labels are arrays now
# it seems no necessary, because model.predict can work with tensors
# but maybe it is used later somewhere
# model.predict returns the numpy array of predictions

print(predictions.shape)  # printing tuple of array dimensions (tuple==cortege)
# should be 32x10, 32 = batch size times 10 predictions for each image

print(predictions[2])  # p-ing the distribution of predictions' probabilities
# for image number 2 in batch number 1

print(np.argmax(predictions[2]))  # printing the number of the array
# with the maximum value among the predictions
print(test_labels[2])  # the same for labels
print(class_names[test_labels[2]])  # printing the name of the label


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

# Plot the  X test images after the shift, their predicted label,
# and the true label
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
print(np.argmax(predictions_single[0]))








