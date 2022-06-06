import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import pandas as pd
import sklearn

# Polynomial Regression: extending linear models with basis functions
'''
One common pattern within machine learning is to use linear models trained on nonlinear functions of the data. 
This approach maintains the generally fast performance of linear methods, while allowing them to fit a much wider range of data.

For example, a simple linear regression can be extended by constructing polynomial features from the coefficients. 
In the standard linear regression case, you might have a model that looks like this for two-dimensional data:
𝑦̂ (𝑤,𝑥)=𝑤0+𝑤1𝑥1+𝑤2𝑥2

If we want to fit a paraboloid to the data instead of a plane, we can combine the features in second-order polynomials, 
so that the model looks like this:
𝑦̂ (𝑤,𝑥)=𝑤0+𝑤1𝑥1+𝑤2𝑥2+𝑤3𝑥1𝑥2+𝑤4𝑥21+𝑤5𝑥22

The (sometimes surprising) observation is that this is still a linear model: to see this, 
imagine creating a new variable
𝑧=[𝑥1,𝑥2,𝑥1𝑥2,𝑥21,𝑥22]

With this re-labeling of the data, our problem can be written
𝑦̂ (𝑤,𝑥)=𝑤0+𝑤1𝑧1+𝑤2𝑧2+𝑤3𝑧3+𝑤4𝑧4+𝑤5𝑧5

We see that the resulting polynomial regression is in the same class of linear models 
we’d considered above (i.e. the model is linear in w) and can be solved by the same techniques. 
By considering linear fits within a higher-dimensional space built with these basis functions, 
the model has the flexibility to fit a much broader range of data.
'''

np.random.seed(42)
n_samples = 100

X = np.linspace(0, 10, 100)
rng = np.random.randn(n_samples) * 100

y = X ** 3 + 100 + rng

plt.figure(figsize=(10, 8))
plt.scatter(X, y)

y = X ** 3 + 100
plt.figure(figsize=(10, 8))
plt.scatter(X, y)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Linear Regression
lr = LinearRegression()
lr.fit(X.reshape(-1, 1), y)
model_pred = lr.predict(X.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(X, y)
plt.plot(X, model_pred)
print(r2_score(y, model_pred))

# Polynomial
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X.reshape(-1, 1))

X[:5]
X_poly[:5]

res = np.array([0, 1, 2, 3, 4, 5])
res

res_two = poly_reg.fit_transform(res.reshape(-1, 1))
res_two

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y.reshape(-1, 1))
y_pred = lin_reg_2.predict(X_poly)

plt.figure(figsize=(10, 8))
plt.scatter(X, y)
plt.plot(X, y_pred)
print(r2_score(y, y_pred))

# Degree 3
degree_three = PolynomialFeatures(degree=3)
example = degree_three.fit_transform(X.reshape(-1, 1))

example[:5]
res_three = degree_three.fit_transform(res.reshape(-1, 1))
res_three


'''
# Boston Housing Dataset
'''
df_boston = pd.read_csv('/Users/yuawong/Documents/GitHub/udemy_repo/The_Complete_Machine_Learning_Course_with_Python/'
                        'data/housing.data', delim_whitespace=True, header=None)
df_boston.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

pd.options.display.float_format = "{:,.2f}".format
df_boston.corr()

X_boston = df_boston['DIS'].values
y_boston = df_boston['NOX'].values

plt.figure(figsize=(12, 8))
plt.scatter(X_boston, y_boston)

# Linear
lr = LinearRegression()
lr.fit(X_boston.reshape(-1, 1), y_boston)
model_pred = lr.predict(X_boston.reshape(-1, 1))

plt.figure(figsize=(12, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_boston, model_pred)
print("R^2 score = {:.2f}".format(r2_score(y_boston, model_pred)))


# Quadratic x^2
poly_reg = PolynomialFeatures(degree=2)
X_poly_b = poly_reg.fit_transform(X_boston.reshape(-1, 1))
lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly_b, y_boston)

X_fit = np.arange(X_boston.min(), X_boston.max(), 1)[:, np.newaxis]
X_fit

y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_fit.reshape(-1, 1)))

plt.figure(figsize=(10, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_fit, y_pred)

print("R^2 score = {:.2f}".format(r2_score(y_boston, lin_reg_2.predict(X_poly_b))))


# Cubic
poly_reg = PolynomialFeatures(degree=3)
X_poly_b = poly_reg.fit_transform(X_boston.reshape(-1, 1))
lin_reg_3 = LinearRegression()

lin_reg_3.fit(X_poly_b, y_boston)

X_fit = np.arange(X_boston.min(), X_boston.max(), 1)[:, np.newaxis]

y_pred = lin_reg_3.predict(poly_reg.fit_transform(X_fit.reshape(-1, 1)))

plt.figure(figsize=(10, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_fit, y_pred)
print("R^2 score = {:.2f}".format(r2_score(y_boston, lin_reg_3.predict(X_poly_b))))


# X^4 Just trying to test things out
poly_reg = PolynomialFeatures(degree=4)
X_poly_b = poly_reg.fit_transform(X_boston.reshape(-1, 1))
lin_reg_4 = LinearRegression()

lin_reg_4.fit(X_poly_b, y_boston)

X_fit = np.arange(X_boston.min(), X_boston.max(), 1)[:, np.newaxis]

y_pred = lin_reg_4.predict(poly_reg.fit_transform(X_fit.reshape(-1, 1)))

plt.figure(figsize=(10, 8))
plt.scatter(X_boston, y_boston)
plt.plot(X_fit, y_pred)
print("R^2 score = {:.2f}".format(r2_score(y_boston, lin_reg_4.predict(X_poly_b))))