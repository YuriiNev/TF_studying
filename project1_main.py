import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

celsius_q = np.array([-40, -10,  0,  8, 15, 22,  38], dtype=float)
fahrenheits_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for (i, c) in enumerate(celsius_q):
    print(i, c)
    print("{} degrees Celsius = {} degrees Fahrenheit"
          .format(c, fahrenheits_a[i]))

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',  # mean_absolute_error
              optimizer=tf.keras.optimizers.Adam(0.1))
print("Finished creating the model")

history1 = model.fit(celsius_q, fahrenheits_a, epochs=1000, verbose=False)
print("Finished training the model")
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history1.history['loss'])
print(model.predict([1000.0]))

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([l0, l1, l2])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history2 = model.fit(celsius_q, fahrenheits_a, epochs=200, verbose=False)
plt.plot(history2.history['loss'])

print("Finished training the model")
print(model.predict([1000.0]))
print("Model predicts that 100 degrees Celsius is: {} degrees \
      Fahrenheit \n \n".format(model.predict([1000.0])))
print("l0 weights {}".format(l0.get_weights()))
print("l1 weights {}".format(l0.get_weights()))
print("l2 weights {}".format(l0.get_weights()))
