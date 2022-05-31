import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from sklearn.datasets import load_boston
boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

X = df
y = boston_data.target

# Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

'''
We need to add a constant term to allow statsmodel.api to calculate the bias / intercepts.
𝑦=𝑚𝑋
𝑦=𝑚𝑋+𝑐
'''

X_constant = sm.add_constant(X)
pd.DataFrame(X_constant)

model = sm.OLS(y, X_constant)

lr = model.fit()

lr.summary()

# Model Statistical Outputs:
'''
Dep. Variable: The dependent variable or target variable
Model: Highlight the model used to obtain this output. It is OLS here. Ordinary least squares / Linear regression
Method: The method used to fit the data to the model. Least squares
No. Observations: The number of observations
DF Residuals: The degrees of freedom of the residuals. Calculated by taking the number of observations less the number of parameters
DF Model: The number of estimated parameters in the model. In this case 13. The constant term is not included.
R-squared: This is the coefficient of determination. Measure of goodness of fit.

𝑅2=1−𝑆𝑆𝑟𝑒𝑠𝑆𝑆𝑡𝑜𝑡
    From wiki,

    The total sum of squares, 𝑆𝑆𝑡𝑜𝑡=∑𝑖(𝑦𝑖−𝑦¯)2

    The regression sum of squares (explained sum of squares), 𝑆𝑆𝑟𝑒𝑔=∑𝑖(𝑓𝑖−𝑦¯)2

    The sum of squares of residuals (residual sum of squares), 𝑆𝑆𝑟𝑒𝑠=∑𝑖(𝑦𝑖−𝑓𝑖)2=∑𝑖𝑒2𝑖

Adj. R-squared: This is the adjusted R-squared. It is the coefficient of determination adjusted by sample size and 
the number of parameters used.
𝑅¯2=1−(1−𝑅2)𝑛−1𝑛−𝑝−1

    𝑝 = The total number of explanatory variables not including the constant term

    𝑛 = The sample size

F-statistic: A measure that tells you if you model is different from a simple average.

Prob (F-statistic): This measures the significance of your F-statistic. Also called p-value of F-statistic. 
In statistics, p-value equal or lower than 0.05 is considered significant.

AIC: This is the Akaike Information Criterion. It evaluatess the model based on the model complexity and 
number of observations. The lower the better.

BIC: This is the Bayesian Information Criterion. Similar to AIC, except it pushishes models with more parameters.
Parameters Estimates and the Associated Statistical Tests

coef: The estimated coefficient. Note that this is just a point estimate.

std err: The standard error of the estimate of the coefficient. Another term for standard deviation

t: The t-statistic score.

P > |t|: The p-value. A measure of the probability that the coefficient is different from zero.

[95.0% Conf. Interval]: The 95% confidence interval of the coefficient. Shown here as [0.025, 0.975], 
the lower and upper bound.
Residual Tests

Omnibus D'Angostino's test: This is a combined statistical test for skewness and kurtosis.

Prob(Omnibus): p-value of Omnibus test.

Skewness: This is a measure of the symmetry of the residuals around the mean. Zero if symmetrical. 
A positive value indicates a long tail to the right; a negative value a long tail to the left.

Kurtosis: This is a measure of the shape of the distribution of the residuals. A normal distribution has a zero measure.
 A negative value points to a flatter than normal distribution; 
 a positive one has a higher peak than normal distribution.

Durbin-Watson: This is a test for the presence of correlation among the residuals. 
This is especially important for time series modelling

Jarque-Bera: This is a combined statistical test of skewness and kurtosis.

Prob (JB): p-value of Jarque-Bera.

Cond. No: This is a test for multicollinearity. > 30 indicates unstable results
'''

# statsmodels.formula.api
form_lr = smf.ols(formula=
                  'y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                  data=df)
mlr = form_lr.fit()

mlr.summary()

form_lr = smf.ols(formula=
                  'y ~ CRIM + ZN + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                  data=df)
mlr = form_lr.fit()
mlr.summary()

# EXERCISE -
# Create a model using the following features: CRIM, ZN, CHAS, NOX
form_lr = smf.ols(formula='y ~ CRIM + ZN + CHAS + NOX', data=df)
mlr = form_lr.fit()
mlr.summary()


# Correlation Matrix
# Useful diagnostic tool to identify collinearity between predictors
pd.options.display.float_format = '{:,.2f}'.format
corr_matrix = df.corr()

corr_matrix[np.abs(corr_matrix) < 0.6] = 0
corr_matrix

plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()


# Detecting Collinearity with Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(df.corr())
pd.options.display.float_format = '{:,.4f}'.format
pd.Series(eigenvalues).sort_values()
# Note that index 8, eigenvalue of 0.0635, is near to zero or very small compared to the others.
# Small value represents presence of collinearity.

np.abs(pd.Series(eigenvectors[:,8])).sort_values(ascending=False)
# Note that index 9, 8, 2 have very high loading when compared against the rest

print(df.columns[2], df.columns[8], df.columns[9])
# Output - NOX, TAX, INDUS
# These are the factors that are causing multicollinearity problem.


# Revisiting Feature Importance and Extractions
'''
Check:
1. Direction of the coefficient
2. Impact of the variable / factor on the model
'''
plt.hist(df['TAX'])

plt.hist(df['NOX'])

# Standardise Variable to Identify Key Feature(s)
'''
In order to perform point 2 properly, one needs to standardise the variable
'''
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

result = pd.DataFrame(list(zip(model.coef_, df.columns)), columns=['coefficient', 'name']).set_index('name')
np.abs(result).sort_values(by='coefficient', ascending=False)


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scaler = StandardScaler()
standard_coefficient_linear_reg = make_pipeline(scaler, model)

standard_coefficient_linear_reg.fit(X,y)
result = pd.DataFrame(list(zip(standard_coefficient_linear_reg.steps[1][1].coef_, df.columns)),
                      columns=['coefficient', 'name']).set_index('name')
np.abs(result).sort_values(by='coefficient', ascending=False)


# Use 𝑅2 to Identify Key Features
'''
- Compare 𝑅2 of model against 𝑅2 of model without a feature.

- A significant change in 𝑅2 signify the importance of the feature.
'''
from sklearn.metrics import r2_score

linear_reg = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                     data=df)
benchmark = linear_reg.fit()
r2_score(y, benchmark.predict(df))

# Without LSTAT
linear_reg = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B',
                     data=df)
benchmark = linear_reg.fit()
r2_score(y, benchmark.predict(df))

# Without Age
linear_reg = smf.ols(formula='y ~ CRIM + ZN + INDUS + CHAS + NOX + RM + DIS + RAD + TAX + PTRATIO + B + LSTAT',
                     data=df)
benchmark = linear_reg.fit()
r2_score(y, benchmark.predict(df))








