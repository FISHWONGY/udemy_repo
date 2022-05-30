# Link to resources - https://docs.google.com/document/d/13VaRATHkUIYXCGxouX3uu5cRgZldwR6-cZdKqQrCpZM/edit#heading=h.hz86pvbn4jty

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("/Users/yuawong/Desktop/pycharmPJ/The_Complete_Machine_Learning_Course_with_Python/data/"
                 "housing.data", delim_whitespace=True, header=None)

# Give name to col.
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

'''
# Or we can load the data from the sklearn lib
from sklearn.datasets import load_boston
boston = load_boston()
df_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
df_boston['MEDV'] = boston.target
'''

# Explore the data
df.describe()

sns.pairplot(df[['CRIM', 'ZN', 'INDUS', 'NOX', 'RM']], height=1.5)
plt.show()

# Correlation Analysis and Feature Selection
pd.options.display.float_format = '{:,.2f}'.format
df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

# Linear Regression with Scikit-Learn
X = df['RM'].values.reshape(-1, 1)
y = df['MEDV'].values
model = LinearRegression()
model.fit(X, y)
model.coef_
model.intercept_

# Plot
plt.figure(figsize=(12, 8));
sns.regplot(X, y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();

# Plot from original df
sns.jointplot(x='RM', y='MEDV', data=df, kind='reg', height=8);
plt.show();

# Predict
model.predict(np.array([7]).reshape(1, -1))

'''
Below is from Jacob T. VanderPlas text, Python Data Science Handbook: Essential Tools for Working with Data

Basics of the API:

Most commonly, the steps in using the Scikit-Learn estimator API are as follows (we will step through a handful of detailed examples in the sections that follow):

    Choose a class of model by importing the appropriate estimator class from Scikit- Learn.
    Choose model hyperparameters by instantiating this class with desired values.
    Arrange data into a features matrix and target vector.
    Fit the model to your data by calling the fit() method of the model instance.
    Apply the model to new data:
        For supervised learning, often we predict labels for unknown data using the predict() method.
        For unsupervised learning, we often transform or infer properties of the data using the transform() or predict() method.
'''

# Step 1: Selecting a model

# Step 2: Instantiation
ml_2 = LinearRegression()

# Step 3: Arrange data
X = df['LSTAT'].values.reshape(-1, 1)
y = df['MEDV'].values

# Step 4: Model fitting
ml_2.fit(X, y)

# Step 5: Predict
ml_2.predict(np.array([15]).reshape(1, -1))

plt.figure(figsize=(12,8));
sns.regplot(X, y);
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.show();


sns.jointplot(x='LSTAT', y='MEDV', data=df, kind='reg', height=8);
plt.show();


# Robust Regression

'''
Outlier Demo: http://digitalfirst.bfwpub.com/stats_applet/stats_applet_5_correg.html

RANdom SAmple Consensus (RANSAC) Algorithm

link = http://scikit-learn.org/stable/modules/linear_model.html#ransac-regression

Each iteration performs the following steps:

Select min_samples random samples from the original data and check whether the set of data is valid (see is_data_valid).

Fit a model to the random subset (base_estimator.fit) and check whether the estimated model is valid (see is_model_valid).

Classify all data as inliers or outliers by calculating the residuals to the estimated model (base_estimator.predict(X) - y) - all data samples with absolute residuals smaller than the residual_threshold are considered as inliers.

Save fitted model as best model if number of inlier samples is maximal. In case the current estimated model has the same number of inliers, it is only considered as the best model if it has better score.
'''

X = df['RM'].values.reshape(-1,1)
y = df['MEDV'].values

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor()

ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

np.arange(3, 10, 1)

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))

sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,8));
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('average number of rooms per dwelling')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper left')
plt.show()

ransac.estimator_.coef_

ransac.estimator_.intercept_

X = df['LSTAT'].values.reshape(-1,1)
y = df['MEDV'].values
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(0, 40, 1)
line_y_ransac = ransac.predict(line_X.reshape(-1, 1))

# Plot
sns.set(style='darkgrid', context='notebook')
plt.figure(figsize=(12,8));
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='brown', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('% lower status of the population')
plt.ylabel("Median value of owner-occupied homes in $1000's")
plt.legend(loc='upper right')
plt.show()


# Performance Evaluation of Regression Model
from sklearn.model_selection import train_test_split
X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lr = LinearRegression()

lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)

# Method 1: Residual Analysis
plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='k')
plt.xlim([-10, 50])
plt.show()

'''
Method 2: Mean Squared Error (MSE)

𝑀𝑆𝐸=1𝑛∑𝑖=1𝑛(𝑦𝑖−𝑦̂ 𝑖)2

The average value of the Sums of Squared Error cost function

Useful for comparing different regression models

For tuning parameters via a grid search and cross-validation
'''

from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, y_train_pred)

mean_squared_error(y_test, y_test_pred)

'''
Method 3: Coefficient of Determination, 𝑅2
𝑅2 = 1−𝑆𝑆𝐸 / 𝑆𝑆𝑇

SSE: Sum of squared errors

SST: Total sum of squares
'''
from sklearn.metrics import r2_score

r2_score(y_train, y_train_pred)

r2_score(y_test, y_test_pred)

# What does a Near Perfect Model Looks like?
generate_random = np.random.RandomState(0)
x = 10 * generate_random.rand(1000)
y = 3 * x + np.random.randn(1000)
plt.figure(figsize = (10, 8))
plt.scatter(x, y);
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X_train.reshape(-1, 1), y_train)

y_train_pred = model.predict(X_train.reshape(-1, 1))
y_test_pred = model.predict(X_test.reshape(-1, 1))

# Method 1: Residual Analysis
plt.figure(figsize=(12,8))
plt.scatter(y_train_pred, y_train_pred - y_train, c='blue', marker='o', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c='orange', marker='*', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-3, xmax=33, lw=2, color='k')
plt.xlim([-5, 35])
plt.ylim([-25, 15])
plt.show()

# Method 2: Mean Squared Error (MSE)
mean_squared_error(y_train, y_train_pred)
mean_squared_error(y_test, y_test_pred)

# Method 3: Coefficient of Determination, 𝑅2
r2_score(y_train, y_train_pred)
r2_score(y_test, y_test_pred)