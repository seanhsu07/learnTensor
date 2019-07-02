import tensorflow as tf
from tensorflow import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

tf.logging.set_verbosity(tf.logging.ERROR)

# setting input and outputs
celcius_q =     np.array([-40, -10,  0, 8, 15, 22, 38], dtype=float)
fahrenheit_a =  np.array([-40,  14, 32, 46, 59, 72, 100], dtype=float)

# create layer
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# create model
model = tf.keras.Sequential([l0])

# compile model
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

# train model
history = model.fit(celcius_q, fahrenheit_a, epochs=500, verbose=False)
print('done')

# plot loss over time
import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

print(model.predict([100.0]))
