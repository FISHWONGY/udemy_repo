import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

x = np.linspace(-6, 6, num=1000)
plt.figure(figsize=(12, 8))
plt.plot(x, 1 / (1 + np.exp(-x))) # Sigmoid Function
plt.title("Sigmoid Function")

tmp = [0, 0.4, 0.6, 0.8, 1.0]
tmp

np.round(tmp)

np.array(tmp) > 0.7


# Making Predictions with Logistic Regression
# Go to solution for 2-class classification problem
'''
𝑦̂ =1.0 / (1.0+𝑒−𝛽0−𝛽1𝑥𝑖)

𝛽0 is the intecept term
𝛽1 is the coefficient for 𝑥𝑖
𝑦̂  is the predicted output with real value between 0 and 1. To convert this to binary output of 0 or 1, 
this would either need to be rounded to an integer value or a cutoff point be provided to specify the class segregation point.
'''

dataset = [[-2.0011, 0],
           [-1.4654, 0],
           [0.0965, 0],
           [1.3881, 0],
           [3.0641, 0],
           [7.6275, 1],
           [5.3324, 1],
           [6.9225, 1],
           [8.6754, 1],
           [7.6737, 1]]

coef = [-0.806605464, 0.2573316]

for row in dataset:
    yhat = 1.0 / (1.0 + np.exp(- coef[0] - coef[1] * row[0]))
    print("yhat {0:.4f}, yhat {1}".format(yhat, round(yhat)))


# Learning with Stochastic Gradient Descent
'''
Logistic Regression uses gradient descent to update the coefficients.

Each gradient descent iteration, the coefficients are updated using the equation:

𝛽=𝛽 + learning rate×(𝑦−𝑦̂ )×𝑦̂ ×(1−𝑦̂ )×𝑥
'''

# Using Scikit Learn to Estimate Coefficients

from sklearn.linear_model import LogisticRegression
dataset

X = np.array(dataset)[:, 0:1]
y = np.array(dataset)[:, 1]

X
y

clf_LR = LogisticRegression(C=1.0, penalty='l2', tol=0.0001, solver="lbfgs")

clf_LR.fit(X, y)

clf_LR.predict(X)

clf_LR.predict_proba(X)


# Classification Exercise
dataset2 = [[0.2,  0.],
            [0.2,  0.],
            [0.2,  0.],
            [0.2,  0.],
            [0.2,  0.],
            [0.4,  0.],
            [0.3,  0.],
            [0.2,  0.],
            [0.2,  0.],
            [0.1,  0.],
            [1.4,  1.],
            [1.5,  1.],
            [1.5,  1.],
            [1.3,  1.],
            [1.5,  1.],
            [1.3,  1.],
            [1.6,  1.],
            [1.,  1.],
            [1.3,  1.],
            [1.4,  1.]]

X = np.array(dataset2)[:, 0:1]
y = np.array(dataset2)[:, 1]

clf_LR = LogisticRegression(C=1.0, penalty='l2', tol=0.0001, solver='lbfgs')

clf_LR.fit(X, y)

y_pred = clf_LR.predict(X)
clf_LR.predict(X)

np.column_stack((y_pred, y))

clf_LR.predict(np.array([0.9]).reshape(1, -1))

clf_LR.predict(np.array([0.4]).reshape(1, -1))

