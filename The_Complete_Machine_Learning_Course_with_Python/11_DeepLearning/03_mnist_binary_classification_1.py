# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import tensorflow as tf

"""## Preparing the Training Data

Load [mnist](http://yann.lecun.com/exdb/mnist/) dataset as distributed with keras

"""

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

tf.keras.backend.image_data_format()

# input image dimensions
img_rows, img_cols = 28, 28

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
if tf.keras.backend.image_data_format() == 'channels_last':
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = np.where(y_train != 5, 0, 1)
# not digit 5, then y target convert to 0
# if it is digit 5, then y target is 1

'''
# If memory not enough, use float32
y_train = np.asarray(y_train.astype('float32'))
y_test = np.asarray(y_test.astype('float32'))
'''

"""## Network Architecture"""

batch_size = 128
num_classes = 1
epochs = 20

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='sigmoid')
])
model.summary()

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy']) # Loss Function and Optimisers

"""## Measure the Performance of the NN"""
# Remaining 50000 used for validation
x_val = x_train[50000:]
# First 50000 used for training
x_train_1 = x_train[:50000]
y_val = y_train[50000:]
y_train_1 = y_train[:50000]

h = model.fit(x_train_1,
              y_train_1,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(x_val, y_val))

"""## Visualisation Performance"""

import matplotlib.pyplot as plt
loss_values = h.history['loss']
val_loss_values = h.history['val_loss']

epochs = range(1, len(h.history['accuracy']) + 1)

plt.style.use('ggplot')
plt.plot(epochs, loss_values, 'bo', 
         label='Training Loss')
plt.plot(epochs, val_loss_values, 'b',
         label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();
# Starting from epoch 10/11 we are starting to overfit our training data

plt.clf()
acc_values = h.history['accuracy']
val_acc_values = h.history['val_accuracy']

plt.plot(epochs, acc_values, 'bo', 
         label='Training Accuracy')
plt.plot(epochs, val_acc_values, 'b',
         label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend();
# Accuracy keeps increasing as epoch increase, which is nice

