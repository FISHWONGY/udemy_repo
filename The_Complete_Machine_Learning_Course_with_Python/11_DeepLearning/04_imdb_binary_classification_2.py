# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

imdb = tf.keras.datasets.imdb
max_features = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(x_train.shape)

"""x_train is columns of features of texts that has already been converted to unique integers. 
The reviews, also called sequences of words, are now sequences of integers. 
Each integer represent a specific word."""

print(x_train[0:2])

"""
0 stands for negative review
1 stands for positive review
"""

print(y_train[0:2])

word_index = imdb.get_word_index()

"""Note that each word is represented by an integer"""

print(word_index)

"""We just need to reverse them by using list / dict comprehension as below:"""

print([(value, key) for (key, value) in word_index.items()])

"""## Data Preparation"""

import numpy as np
# from keras.preprocessing import sequence

"""
Convert the x_train array from a features of text (integers) to a matrix of 0 and 1. 

0 - when no such word exist. 
1 - when the word exist in the slice of data.
"""


# For one hot encoding for X
def matrix_seq(data, dim=max_features):
    results = np.zeros((len(data), dim))    # create a matrix of zeros for writing
    for i, loc in enumerate(data):
        results[i, loc] = 1.                # Put a 1.0 when the word exist
    return results


x_train = matrix_seq(x_train)
x_test = matrix_seq(x_test)

y_train = np.asarray(y_train.astype('float32'))
y_test = np.asarray(y_test.astype('float32'))

print(y_train.shape)

"""## Network Architecture"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5000,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')])
model.summary()

model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

"""What are activation functions?

`relu`: non-linear. So the NN model can learn non-linear features

Else it will be purely linear (affine) in nature

## Measure the Performance of the NN

Let's use some of the training data for validation
"""
# For validation - test
x_val = x_train[:10000]
# Training set
x_train_1 = x_train[10000:]

y_val = y_train[:10000]
y_train_1 = y_train[10000:]

h = model.fit(x_train_1,
              y_train_1,
              epochs=50,
              batch_size=512,
              validation_data=(x_val, y_val))

"""## Visualisation Performance"""

import matplotlib.pyplot as plt
# plotting for loss
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
# After Epoch 25, over-fitting happening

# Ploting for accuracy
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

