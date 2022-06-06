# # Keras Basics
# 
# Welcome to the section on deep learning! We'll be using Keras with a TensorFlow backend to perform our deep learning operations.
# This means we should get familiar with some Keras fundamentals and basics!

import numpy as np
# ## Dataset
# 
# We will use the famous Iris Data set.
from sklearn.datasets import load_iris
iris = load_iris()
type(iris)
print(iris.DESCR)
X = iris.data
print(X)

y = iris.target

print(y)

from keras.utils import to_categorical
y = to_categorical(y)

y.shape

# ## Split the Data into Training and Test
# 
# Its time to split the data into a train/test set. Keep in mind, sometimes people like to split 3 ways, train/test/validation. We'll keep things simple for now. **Remember to check out the video explanation as to why we split and what all the parameters mean!**
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(X_train)

print(X_test)
print(y_train)
print(y_test)


# ## Standardizing the Data
# Usually when using Neural Networks, you will get better performance when you standardize the data. Standardization just means normalizing the values to all fit between a certain range, like 0-1, or -1 to 1.
# The scikit learn library also provides a nice function for this.

from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)


# Ok, now we have the data scaled!
X_train.max()

scaled_X_train.max()
X_train

scaled_X_train


# ## Building the Network with Keras
# 
# Let's build a simple neural network!
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# ## Fit (Train) the Model
# Play around with number of epochs as well!
model.fit(scaled_X_train, y_train, epochs=150, verbose=2)


# ## Predicting New Unseen Data
# 
# Let's see how we did by predicting on **new data**. Remember, our model has **never** seen the test data that we scaled previously! This process is the exact same process you would use on totally brand new data. For example , a brand new bank note that you just analyzed .
scaled_X_test

# Spits out probabilities by default.
# model.predict(scaled_X_test)

model.predict_classes(scaled_X_test)


# # Evaluating Model Performance
# 
# So how well did we do? How do we actually measure "well". Is 95% accuracy good enough? It all depends on the situation. Also we need to take into account things like recall and precision. Make sure to watch the video discussion on classification evaluation before running this code!

model.metrics_names


# In[140]:
model.evaluate(x=scaled_X_test, y=y_test)

from sklearn.metrics import confusion_matrix, classification_report

predictions = model.predict_classes(scaled_X_test)

predictions

y_test.argmax(axis=1)

confusion_matrix(y_test.argmax(axis=1), predictions)

print(classification_report(y_test.argmax(axis=1), predictions))


# ## Saving and Loading Models
# 
# Now that we have a model trained, let's see how we can save and load it.

model.save('myfirstmodel.h5')

from keras.models import load_model

newmodel = load_model('myfirstmodel.h5')

newmodel.predict_classes(X_test)


# Great job! You now know how to preprocess data, train a neural network, and evaluate its classification performance!
