import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import sklearn

from sklearn.datasets import load_boston
boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

X = df[['LSTAT']].values
y = boston_data.target

plt.figure(figsize=(8, 6))
plt.scatter(X, y)

# Without Pre-processing
alpha = 0.0001
w_ = np.zeros(1 + X.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X, w_[1:]) + w_[0]
    errors = (y - y_pred)

    w_[1:] += alpha * X.T.dot(errors)
    w_[0] += alpha * errors.sum()

    cost = (errors ** 2).sum() / 2.0
    cost_.append(cost)

plt.figure(figsize=(8, 6))
plt.plot(range(1, n_ + 1), cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')


# With Pre-processing
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y.reshape(-1,1)).flatten()

alpha = 0.0001
w_ = np.zeros(1 + X_std.shape[1])
cost_ = []
n_ = 100

for i in range(n_):
    y_pred = np.dot(X_std, w_[1:]) + w_[0]
    errors = (y_std - y_pred)

    w_[1:] += alpha * X_std.T.dot(errors)
    w_[0] += alpha * errors.sum()

    cost = (errors ** 2).sum() / 2.0
    cost_.append(cost)
plt.figure(figsize=(8, 6))
plt.plot(range(1, n_ + 1), cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')


# Before Scaling
plt.figure(figsize=(8, 6))
plt.hist(X)
plt.xlim(-40, 40)

# After Scaling
plt.figure(figsize=(8, 6))
plt.hist(X_std)
plt.xlim(-4, 4)


# Data Pre-processing
'''
- Standardization / Mean Removal

- Min-Max or Scaling Features to a Range

- Normalization

- Binarization

Assumptions:

- Implicit/explicit assumption of machine learning algorithms: The features follow a normal distribution.
- Most method are based on linear assumptions
- Most machine learning requires the data to be standard normally distributed. Gaussian with zero mean and unit variance

In practice we often ignore the shape of the distribution and just transform the data to center it by 
removing the mean value of each feature, then scale it by dividing non-constant features by their standard deviation.

For instance, many elements used in the objective function of a learning algorithm 
(such as the RBF kernel of Support Vector Machines or the l1 and l2 regularizers of linear models) assume that 
all features are centered around zero and have variance in the same order. 
If a feature has a variance that is orders of magnitude larger than others, 
it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
'''

from sklearn import preprocessing
X_train = np.array([[1., -1.,  2.],
                    [2.,  0.,  0.],
                    [0.,  1., -1.]])

X_train.mean(axis=0)


# Standardization / Mean Removal / Variance Scaling
'''
Mean is removed. Data is centered on zero. This is to remove bias.
Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; 
they might behave badly if the individual features do not more or less look like standard normally distributed data: 
Gaussian with zero mean and unit variance. "standard normal" random variable with mean 0 and standard deviation 1.

𝑋′=𝑋−𝑋¯𝜎
'''
X_scaled = preprocessing.scale(X_train)
X_scaled

# Scaled data has zero mean and unit variance (unit variance means variance = 1):
X_scaled.mean(axis=0)

X_scaled.std(axis=0)
# Keeping in mind that if you have scaled your training data,
# you must do likewise with your test data as well. However,
# your assumption is that the mean and variance must be invariant between your train and test data.
# scikit-learn assists with a built-in utility function StandardScaler.

scaler = preprocessing.StandardScaler().fit(X_train)
scaler

scaler.mean_
scaler.scale_
scaler.transform(X_train)

plt.figure(figsize=(8, 6))
plt.hist(X_train)
# You can now utilise the transform for new dataset

X_test = [[-1., 1., 0.]]
scaler.transform(X_test)


# Min-Max or Scaling Features to a Range
'''
Scaling features to lie between a given minimum and maximum value, often between zero and one, 
or so that the maximum absolute value of each feature is scaled to unit size.

The motivation to use this scaling include robustness to very small standard deviations of features and 
preserving zero entries in sparse data.
'''

# MinMaxScaler
# Scale a data to the [0, 1] range:
X_train = np.array([[1., -1.,  2.],
                    [2.,  0.,  0.],
                    [0.,  1., -1.]])

min_max_scaler = preprocessing.MinMaxScaler()

X_train_minmax = min_max_scaler.fit_transform(X_train)

X_train_minmax

# Now to unseen data
X_test = np.array([[-3., -1.,  0.], [2., 1.5, 4.]])
X_test_minmax = min_max_scaler.transform(X_test)
X_test_minmax

'''
doc:
Init signature: preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)

Transforms features by scaling each feature to a given range.

This estimator scales and translates each feature individually such that it is in the given range on the training set, 
i.e. between zero and one.

The transformation is given by::
'''

X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max-min) + min

'''
𝑋𝑠𝑡𝑑=𝑋−𝑋𝑚𝑖𝑛𝑋𝑚𝑎𝑥−𝑋𝑚𝑖𝑛

𝑋′=𝑋𝑠𝑡𝑑(max−min)+min
'''

# MaxAbsScaler
'''
Works in a very similar fashion, but scales in a way that the training data lies within the range [-1, 1] by 
dividing through the largest maximum value in each feature. 
It is meant for data that is already centered at zero or sparse data.
'''
X_train = np.array([[1., -1.,  2.],
                    [2.,  0.,  0.],
                    [0.,  1., -1.]])

max_abs_scaler = preprocessing.MaxAbsScaler()
X_train_maxabs = max_abs_scaler.fit_transform(X_train)
X_train_maxabs

X_test = np.array([[ -1., -0.5,  2.], [0., 0.5, -0.6]])
X_test_maxabs = max_abs_scaler.transform(X_test)
X_test_maxabs

# Scaling sparse data
'''
Centering sparse data would destroy the sparseness structure in the data, and thus rarely is a sensible thing to do.

However, it can make sense to scale sparse inputs, especially if features are on different scales.

MaxAbsScaler and maxabs_scale were specifically designed for scaling sparse data
'''

# Scaling vs Whitening
'''
It is sometimes not enough to center and scale the features independently, 
since a downstream model can further make some assumption on the linear independence of the features.

To address this issue you can use sklearn.decomposition.PCA or 
sklearn.decomposition.RandomizedPCA with whiten=True to further remove the linear correlation across features.
'''

# Normalization
'''
Normalization is the process of scaling individual samples to have unit norm.

This process can be useful if you plan to use a quadratic form such as the dot-product or 
any other kernel to quantify the similarity of any pair of samples.

𝑋′=𝑋−𝑋𝑚𝑒𝑎𝑛𝑋𝑚𝑎𝑥−𝑋𝑚𝑖𝑛

This assumption is the base of the Vector Space Model often used in text classification and clustering contexts.

There are two types of Normalization

1. L1 normalization, Least Absolute Deviations Ensure the sum of absolute values is 1 in each row.

2. L2 normalization, Least squares, Ensure that the sum of squares is 1.
'''

X = [[1., -1.,  2.],
     [2.,  0.,  0.],
     [0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')

X_normalized

# Alternatively
# The preprocessing module further provides a utility class Normalizer that
# implements the same operation using the Transformer API.
normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer

normalizer.transform(X)

normalizer.transform([[-1.,  1., 0.]])

# Binarization
'''
𝑓(𝑥)=0,1

Feature binarization is the process of thresholding numerical features to get boolean values. 
This can be useful for downstream probabilistic estimators that make assumption that the input data is 
distributed according to a multi-variate Bernoulli distribution

It is also common among the text processing community to use binary feature values 
(probably to simplify the probabilistic reasoning) even if normalized counts (a.k.a. term frequencies) or 
TF-IDF valued features often perform slightly better in practice.
'''
X = [[1., -1.,  2.],
     [2.,  0.,  0.],
     [0.,  1., -1.]]

binarizer = preprocessing.Binarizer().fit(X)  # fit does nothing
binarizer

binarizer.transform(X)

# Modifying the threshold
binarizer = preprocessing.Binarizer(threshold=-0.5)

binarizer.transform(X)

'''
# Encoding categorical features
'''
source = ['australia', 'singapore', 'new zealand', 'hong kong']

label_enc = preprocessing.LabelEncoder()
src = label_enc.fit_transform(source)

print("country to code mapping:\n")
for k, v in enumerate(label_enc.classes_):
    print(v, '\t', k)


test_data = ['hong kong', 'singapore', 'australia', 'new zealand']
result = label_enc.transform(test_data)
print(result)

# One Hot / One-of-K Encoding
'''
- Useful for dealing with sparse matrix
- uses one-of-k scheme

The process of turning a series of categorical responses into a set of binary result (0 or 1)
'''
from sklearn.preprocessing import OneHotEncoder
one_hot_enc = OneHotEncoder(sparse=False, categories='auto')
src = src.reshape(len(src), 1)
one_hot = one_hot_enc.fit_transform(src)
print(one_hot)

invert_res = label_enc.inverse_transform([np.argmax(one_hot[0, :])])
print(invert_res)

invert_res = label_enc.inverse_transform([np.argmax(one_hot[3, :])])
print(invert_res)

