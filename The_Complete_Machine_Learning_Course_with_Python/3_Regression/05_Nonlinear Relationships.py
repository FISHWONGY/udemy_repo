import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import sklearn
from sklearn.datasets import load_boston

boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

y = boston_data.target

'''
# 1st Regressor
# Brief Introduction to Decision Tree
'''
from sklearn.tree import DecisionTreeRegressor
X = df[['LSTAT']].values

# Parameter - max_depth we need to decide
# No way to know if it's too deep, we just need to try and plot things out to see
tree = DecisionTreeRegressor(max_depth=5)
tree.fit(X, y)

sort_idx = X.flatten().argsort()
plt.figure(figsize=(10, 8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color='k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV')


# Using max_depth of 5 led to overfitting. Let's try 2 instead.
tree = DecisionTreeRegressor(max_depth=2)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
plt.figure(figsize=(10, 8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], tree.predict(X[sort_idx]), color='k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV')

# 3 works the best, 4 begins to over-fitting

# Brief Introduction to Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = df.values
# y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=500, criterion='mse',
                               random_state=42, n_jobs=-1)

forest.fit(X_train, y_train)

y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print("MSE train: {0:.4f}, test: {1:.4f}".format(mean_squared_error(y_train, y_train_pred),
                                                 mean_squared_error(y_test, y_test_pred)))

print("R^2 train: {0:.4f}, test: {1:.4f}".format(r2_score(y_train, y_train_pred),
                                                 r2_score(y_test, y_test_pred)))

'''
# Brief Introduction to AdaBoost
'''
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                        n_estimators=500, random_state=42)

ada.fit(X_train, y_train)

y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

print("MSE train: {0:.4f}, test: {1:.4f}".format(mean_squared_error(y_train, y_train_pred),
                                                 mean_squared_error(y_test, y_test_pred)))

print("R^2 train: {0:.4f}, test: {1:.4f}".format(r2_score(y_train, y_train_pred),
                                                 r2_score(y_test, y_test_pred)))

# Random Forest works better than AdaBoost in this case

'''
Revisiting Feature Importance

13 features.
Are they all equally important?
Which features are more important?
Can scikit-learn help us with this?
'''

# According to AdaBoost - which features are more important
# LSTAT, RM, DIS, RATIO,NOX, CRIM
result = pd.DataFrame(ada.feature_importances_, df.columns)
result.columns = ['feature']
result.sort_values(by='feature', ascending=False)

result.sort_values(by='feature', ascending=False).plot(kind='bar')

# According to Random Forest
# LSTAT, RM, DIS, CRIM, RATIO, NOX
result = pd.DataFrame(forest.feature_importances_, df.columns)
result.columns = ['feature']
result.sort_values(by='feature', ascending=False).plot(kind='bar')

'''
Exercise

Time for you to try your hands on building machine learning model. All the codes have been provided. 
All you have to do is study the codes and make some light modifications to tacklet the problem that I am presenting to you now.

In previous and this lesson, I performed modelling of our data using Random Forest and AdaBoost. 
I would like you to make use of Decision Tree Regressor and perform the following:

1. Split your data to train and test set, reserving 30% of your data for testing
2. Instantiate, fit and predict
3. Calculate MSE and R-square
4. Extract feature importance and visualise it using bar chart
'''
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

print("MSE train: {0:.4f}, test: {1:.4f}".format(mean_squared_error(y_train, y_train_pred),
                                                 mean_squared_error(y_test, y_test_pred)))

print("R^2 train: {0:.4f}, test: {1:.4f}".format(r2_score(y_train, y_train_pred),
                                                 r2_score(y_test, y_test_pred)))

# According to Decision Tree Regressor
result = pd.DataFrame(tree.feature_importances_, df.columns)
result.columns = ['feature']
result.sort_values(by='feature', ascending=False).plot(kind='bar')


'''
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X_train, y_train)

y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

print("MSE train: {0:.4f}, test: {1:.4f}".format(mean_squared_error(y_train, y_train_pred),
                                                 mean_squared_error(y_test, y_test_pred)))

print("R^2 train: {0:.4f}, test: {1:.4f}".format(r2_score(y_train, y_train_pred),
                                                 r2_score(y_test, y_test_pred)))

result = pd.DataFrame(tree.feature_importances_, df.columns)
result.columns = ['feature']
result.sort_values(by='feature', ascending=False).plot(kind='bar')
'''

