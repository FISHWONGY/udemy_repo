# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf

"""## Preparing the Training Data

Load [mnist](http://yann.lecun.com/exdb/mnist/) dataset as distributed with keras

"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
print(np.unique(y_train))
print(np.unique(y_test))

import matplotlib.pyplot as plt
plt.imshow(x_train[0], cmap='Greys')

print(y_train[0])

"""
Data preparation:
* the data came in the form of `uint8` with value in the `[0, 255]` range. We need to transform it into Python `float32` array with values between 0 and 1."""

x_train, x_test = x_train / 255.0, x_test / 255.0

"""## Defining, Compiling and Fitting Our Model

### Model 1:

Our network has three layers:

* **First Layer: input** `tf.keras.layers.Flatten` — 
This layer flattens the digit images from a 2d-array of 28 $\times$ 28 pixels, 
to a 1d-array of 784 pixels (28\*28). This layer only reformats the data.

* **Second Layer: "hidden"** `tf.keras.layers.Dense`— A densely connected layer of 512 neurons. 
Each neuron (or node) takes input from all 784 nodes in the previous layer. Recall that we flattened the image. 


* **output** `tf.keras.layers.Dense` — A 10-node *softmax* layer, 
with each node representing each of the digit class. 
As in the previous layer, each node takes input from the 512 nodes in the layer before it. 
Each node weights the input according to learned parameters, and then outputs a value in the range `[0, 1]`, 
representing the probability that the image belongs to that class. The sum of all 10 node values is 1.
"""

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=10, activation='softmax')
])
model.summary()
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
h = model.fit(x_train, y_train, epochs=10, batch_size=256)

"""The `batch_size` tells `model.fit` to update the model variables after every batches of 256 images."""

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(h.history['loss'])

eval_loss, eval_acc = model.evaluate(x_test, 
                                     y_test)
print('Test accuracy: {}'.format(eval_acc))
# Test accuracy: 0.9797999858856201
# This model is a bit over-trained & overfitting, since the accuracy @ epoch 10 uis way higher than print('Test accuracy: {}'.format(eval_acc))
"""### Model 2"""

model2 = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model2.summary()
model2.compile(optimizer='adam',
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])
h = model2.fit(x_train, y_train,
               epochs=10,
               batch_size=256)

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(h.history['loss'])

eval_loss, eval_acc = model2.evaluate(x_test,
                                      y_test)
print('Test accuracy: {}'.format(eval_acc))
# Test accuracy: 0.9765999913215637
# This model is not over-train/over fit, since its accuracy @ epoxh 10 is similar to print('Test accuracy: {}'.format(eval_acc))

"""***"""

img = x_test[0]
print(img.shape)
img = np.array([img])
print(img.shape)

prediction = model2.predict(img)
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
print(prediction * 100)
# 99.86% that this img is 7

np.argmax(prediction[0])
# Prediction - 7

print(y_test[0])
# Actual label - 7
"""***"""