# Regularized Method for Regression

import pandas as pd
import seaborn as sns

# 3 Types of regularisation
'''
Ridge Regression

Ridge regression addresses some of the problems of Ordinary Least Squares by imposing a penalty on the size of coefficients. 
The ridge coefficients minimize a penalized residual sum of squares,

min𝑤∣∣∣∣𝑋𝑤−𝑦∣∣∣∣22+𝛼∣∣∣∣𝑤∣∣∣∣22

𝛼>=0 is a complexity parameter that controls the amount of shrinkage: 
the larger the value of 𝛼, the greater the amount of shrinkage and 
thus the coefficients become more robust to collinearity.

Ridge regression is an L2 penalized model. Add the squared sum of the weights to the least-squares cost function.

Shows the effect of collinearity in the coefficients of an estimator.

Ridge Regression is the estimator used in this example. 
Each color represents a different feature of the coefficient vector, 
and this is displayed as a function of the regularization parameter.

This example also shows the usefulness of applying Ridge regression to highly ill-conditioned matrices. 
For such matrices, a slight change in the target variable can cause huge variances in the calculated weights. 
In such cases, it is useful to set a certain regularization (alpha) to reduce this variation (noise).
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import sklearn

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

# ###########################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)

coefs = []
for a in alphas:
    ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

# ###########################################################################
# Display results

plt.figure(figsize=(10, 8))
ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

# LASSO Regression
'''
A linear model that estimates sparse coefficients.

Mathematically, it consists of a linear model trained with ℓ1 prior as regularizer. 
The objective function to minimize is:

min𝑤12𝑛𝑠𝑎𝑚𝑝𝑙𝑒𝑠∣∣∣∣𝑋𝑤−𝑦∣∣∣∣22+𝛼∣∣∣∣𝑤∣∣∣∣1

The lasso estimate thus solves the minimization of the least-squares penalty with 𝛼∣∣∣∣𝑤∣∣∣∣1 added, 
where 𝛼 is a constant and ∣∣∣∣𝑤∣∣∣∣1 is the ℓ1−𝑛𝑜𝑟𝑚 of the parameter vector.
'''

# Elastic Net
'''
A linear regression model trained with L1 and L2 prior as regularizer.

This combination allows for learning a sparse model where few of the weights are non-zero like Lasso, 
while still maintaining the regularization properties of Ridge.

Elastic-net is useful when there are multiple features which are correlated with one another. 
Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.

A practical advantage of trading-off between Lasso and Ridge 
is it allows Elastic-Net to inherit some of Ridge’s stability under rotation.

The objective function to minimize is in this case

min𝑤12𝑛𝑠𝑎𝑚𝑝𝑙𝑒𝑠∣∣∣∣𝑋𝑤−𝑦∣∣∣∣22+𝛼𝜌∣∣∣∣𝑤∣∣∣∣1+𝛼(1−𝜌)2∣∣∣∣𝑤∣∣∣∣22
'''

# Outliers Impact

# Linear Regression
from sklearn.linear_model import LinearRegression
np.random.seed(42)
n_samples = 100
rng = np.random.randn(n_samples) * 10
y_gen = 0.5 * rng + 2 * np.random.randn(n_samples)

lr = LinearRegression()
lr.fit(rng.reshape(-1, 1), y_gen)
model_pred = lr.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, model_pred)
print("Coefficient Estimate: ", lr.coef_)


idx = rng.argmax()
y_gen[idx] = 200
idx = rng.argmin()
y_gen[idx] = -200


plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)

o_lr = LinearRegression(normalize=True)
o_lr.fit(rng.reshape(-1, 1), y_gen)
o_model_pred = o_lr.predict(rng.reshape(-1, 1))

plt.scatter(rng, y_gen)
plt.plot(rng, o_model_pred)
print("Coefficient Estimate: ", o_lr.coef_)


# Ridge Regression
from sklearn.linear_model import Ridge
ridge_mod = Ridge(alpha=0.5, normalize=True)
ridge_mod.fit(rng.reshape(-1, 1), y_gen)
ridge_model_pred = ridge_mod.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, ridge_model_pred)

print("Coefficient Estimate: ", ridge_mod.coef_)


# Lasso Regression
from sklearn.linear_model import Lasso
lasso_mod = Lasso(alpha=0.4, normalize=True)
lasso_mod.fit(rng.reshape(-1, 1), y_gen)
lasso_model_pred = lasso_mod.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, lasso_model_pred)

print("Coefficient Estimate: ", lasso_mod.coef_)


# Elastic Net Regression
from sklearn.linear_model import ElasticNet
en_mod = ElasticNet(alpha=0.02, normalize=True)
en_mod.fit(rng.reshape(-1, 1), y_gen)
en_model_pred = en_mod.predict(rng.reshape(-1, 1))

plt.figure(figsize=(10, 8))
plt.scatter(rng, y_gen)
plt.plot(rng, en_model_pred)

print("Coefficient Estimate: ", en_mod.coef_)


# SUMMARY
'''
When should I use Lasso, Ridge or Elastic Net?

Ridge regression can't zero out coefficients; 
You either end up including all the coefficients in the model, or none of them.

LASSO does both parameter shrinkage and variable selection automatically.

If some of your covariates are highly correlated, you may want to look at the Elastic Net instead of the LASSO.
'''

