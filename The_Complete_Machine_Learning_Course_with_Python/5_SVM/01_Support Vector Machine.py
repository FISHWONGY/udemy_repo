import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
sns.set_style('whitegrid')

# Linear SVM Classification
'''
 - Support Vectors
 - Separate with a straight line (linearly separable)
 - Margin
    Hard margin classification
        - Strictly based on those that are at the margin between the two classes
        - However, this is sensitive to outliers
    Soft margin classification
        - Widen the margin and allows for violation
        - With Python Scikit-Learn, you control the width of the margin
        - Control with C hyperparameter
            - smaller C leads to a wider street but more margin violations
             - High C - fewer margin violations but ends up with a smaller margin
'''
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
df = sns.load_dataset('iris')

col = ['petal_length', 'petal_width', 'species']
df.loc[:, col].head()
df.species.unique()

col = ['petal_length', 'petal_width']
X = df.loc[:, col]

species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']

'''
LinearSVC
    - Similar to SVC with parameter kernel=’linear’, but implemented in terms of liblinear rather than libsvm, 
      so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.

SVC
    - C-Support Vector Classification.
    - The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples 
      which makes it hard to scale to dataset with more than a couple of 10000 samples.
'''

C = 0.001
clf = svm.SVC(kernel='linear', C=C)
# clf = svm.LinearSVC(C=C, loss='hinge')
# clf = svm.SVC(kernel='poly', degree=3, C=C)
# clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
clf.fit(X, y)

# Predicting 'petal_length' = 6 and 'petal_width' = 2, output = 2 = virginica
clf.predict([[6, 2]])

# plot the above result out
Xv = X.values.reshape(-1, 1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8, 6))
ax = plt.contourf(xx, yy, z, cmap='afmhot', alpha=0.3)
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80,
            alpha=0.9, edgecolors='g')


###
# Linear SVM Implementation
df = sns.load_dataset('iris')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
col = ['petal_length', 'petal_width']
X = df.loc[:, col]
species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=0)

# Scale Features
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)

C = 1.0 #0.01
clf = svm.SVC(kernel='linear', C=C)
clf.fit(X_std_train, y_train)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# Cross Validation within Train Dataset
res = cross_val_score(clf, X_std_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

y_train_pred = cross_val_predict(clf, X_std_train, y_train, cv=3)

confusion_matrix(y_train, y_train_pred)

print("Precision Score: \t {0:.4f}".format(precision_score(y_train,
                                                           y_train_pred,
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train,
                                                       y_train_pred,
                                                       average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train,
                                               y_train_pred,
                                               average='weighted')))
# The above confusionn metrix is perform with training data, it is supposed to perform better anyway


# Cross Validation within Test Dataset
y_test_pred = cross_val_predict(clf, sc_x.transform(X_test), y_test, cv=3)
confusion_matrix(y_test, y_test_pred)

print("Precision Score: \t {0:.4f}".format(precision_score(y_test,
                                                           y_test_pred,
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test,
                                                       y_test_pred,
                                                       average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test,
                                               y_test_pred,
                                               average='weighted')))


'''
###
# 2. Polynomial Kernel
'''
C = 1.0
clf = svm.SVC(kernel='poly', degree=3, C=C, gamma='auto')
clf.fit(X, y)

Xv = X.values.reshape(-1, 1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8, 6))
ax = plt.contourf(xx, yy, z, cmap='afmhot', alpha=0.3)
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80,
            alpha=0.9, edgecolors='g')


# Polynomial SVM Implementation
df = sns.load_dataset('iris')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
col = ['petal_length', 'petal_width']
X = df.loc[:, col]
species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']
X_train, X_std_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8,
                                                        random_state=0)

# Scale Features
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)

C = 1.0 # 0.01
clf = svm.SVC(kernel='poly', degree=10, C=C, gamma='auto') # 5
clf.fit(X_std_train, y_train)


# Cross Validation within Train Dataset
res = cross_val_score(clf, X_std_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
# It's worst than the linear model above
# Linear model: 0.9x, polynomial  model 0.78


y_train_pred = cross_val_predict(clf, X_std_train, y_train, cv=3)

confusion_matrix(y_train, y_train_pred)

print("Precision Score: \t {0:.4f}".format(precision_score(y_train,
                                                           y_train_pred,
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train,
                                                       y_train_pred,
                                                       average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train,
                                               y_train_pred,
                                               average='weighted')))
# Almost 20% drop compare to the linear model above


# Cross Validation within Test Dataset
y_test_pred = cross_val_predict(clf, sc_x.transform(X_test), y_test, cv=3)
confusion_matrix(y_test, y_test_pred)

print("Precision Score: \t {0:.4f}".format(precision_score(y_test,
                                                           y_test_pred,
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test,
                                                       y_test_pred,
                                                       average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test,
                                               y_test_pred,
                                               average='weighted')))
# 0.6 compare to 0.9 with the linear model


###
# 3. Gaussian Radial Basis Function (rbf)
'''
The kernel function can be any of the following:
    - linear: ⟨𝑥,𝑥′⟩.
    
    - polynomial: (𝛾⟨𝑥,𝑥′⟩+𝑟)𝑑.
    𝑑 is specified by keyword degree
    𝑟 by coef0.
    
    - rbf: exp(−𝛾‖𝑥−𝑥′‖2).
    𝛾 is specified by keyword gamma must be greater than 0.
    
    - sigmoid (tanh(𝛾⟨𝑥,𝑥′⟩+𝑟))
    where 𝑟 is specified by coef0.
'''
df = sns.load_dataset('iris')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
col = ['petal_length', 'petal_width']
X = df.loc[:, col]
species_to_num = {'setosa': 0,
                  'versicolor': 1,
                  'virginica': 2}
df['tmp'] = df['species'].map(species_to_num)
y = df['tmp']
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=0) #0.6


# Scale Features
# SVM is really sensity to this, so usually when using SVM, we'll need to scale X
sc_x = StandardScaler()
X_std_train = sc_x.fit_transform(X_train)


C = 1.0
# Specify rbf
clf = svm.SVC(kernel='rbf', gamma=0.7, C=C)
clf.fit(X_std_train, y_train)


# Cross Validation within Train Dataset
res = cross_val_score(clf, X_std_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))

y_train_pred = cross_val_predict(clf, X_std_train, y_train, cv=3)

confusion_matrix(y_train, y_train_pred)

print("Precision Score: \t {0:.4f}".format(precision_score(y_train,
                                                           y_train_pred,
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train,
                                                       y_train_pred,
                                                       average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train,
                                               y_train_pred,
                                               average='weighted')))
# Radial Basis Function (rbf) performs better than poly, slightly better compare to linear


# Grid Search
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
pipeline = Pipeline([('clf', svm.SVC(kernel='rbf', C=1, gamma=0.1))])

params = {'clf__C': (0.1, 0.5, 1, 2, 5, 10, 20),
          'clf__gamma': (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1)}

svm_grid_rbf = GridSearchCV(pipeline, params, n_jobs=-1,
                            cv=3, verbose=1, scoring='accuracy')

svm_grid_rbf.fit(X_train, y_train)

svm_grid_rbf.best_score_

best = svm_grid_rbf.best_estimator_.get_params()

for k in sorted(params.keys()):
    print('\t{0}: \t {1:.2f}'.format(k, best[k]))


# Conducting validation in the test dataset.
y_test_pred = svm_grid_rbf.predict(X_test)

confusion_matrix(y_test, y_test_pred)

print("Precision Score: \t {0:.4f}".format(precision_score(y_test,
                                                           y_test_pred,
                                                           average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test,
                                                       y_test_pred,
                                                       average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test,
                                               y_test_pred,
                                               average='weighted')))

Xv = X.values.reshape(-1, 1)
h = 0.02
x_min, x_max = Xv.min(), Xv.max() + 1
y_min, y_max = y.min(), y.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

z = svm_grid_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
fig = plt.figure(figsize=(8, 6))
ax = plt.contourf(xx, yy, z, cmap='afmhot', alpha=0.3)
plt.scatter(X.values[:, 0], X.values[:, 1], c=y, s=80,
            alpha=0.9, edgecolors='g')


###
# 4. Support Vector Regression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import pandas as pd
from sklearn.svm import SVR # SVR = Support Vector Regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_boston
boston_data = load_boston()
df = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

y = boston_data.target
X = df[['LSTAT']].values

svr = SVR(gamma='auto')
svr.fit(X, y)

sort_idx = X.flatten().argsort()

plt.figure(figsize=(10, 8))
plt.scatter(X[sort_idx], y[sort_idx])
plt.plot(X[sort_idx], svr.predict(X[sort_idx]), color='k')

plt.xlabel('LSTAT')
plt.ylabel('MEDV')

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Linear Kernel
svr = SVR(kernel='linear')
svr.fit(X_train, y_train)

y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# MSE train: 41.8187, test: 36.8372
print("MSE train: {0:.4f}, test: {1:.4f}".\
      format(mean_squared_error(y_train, y_train_pred),
             mean_squared_error(y_test, y_test_pred)))

# R^2 train: 0.5242, test: 0.5056
print("R^2 train: {0:.4f}, test: {1:.4f}".\
      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))


# Polynomial
svr = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
svr.fit(X_train, y_train)

y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# MSE train: 110.3334, test: 114.3676
print("MSE train: {0:.4f}, test: {1:.4f}".\
      format(mean_squared_error(y_train, y_train_pred),
             mean_squared_error(y_test, y_test_pred)))

# R^2 train: -0.2553, test: -0.5349
# R^2 shd never be -ve, poly model shouldn't be use here
print("R^2 train: {0:.4f}, test: {1:.4f}".\
      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))


# rbf Kernel
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(X_train, y_train)

y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# MSE train: 27.5635, test: 26.7051
print("MSE train: {0:.4f}, test: {1:.4f}".\
      format(mean_squared_error(y_train, y_train_pred),
             mean_squared_error(y_test, y_test_pred)))

# R^2 train: 0.6864, test: 0.6416
print("R^2 train: {0:.4f}, test: {1:.4f}".\
      format(r2_score(y_train, y_train_pred),
             r2_score(y_test, y_test_pred)))

'''
PERFOMANCE CONCLUSION - Again
rbf performs the best, follow by linear than poly
'''
#

###
# 5. Advantages and Disadvantages
'''
The advantages of support vector machines are:
    - Effective in high dimensional spaces.
    - Uses only a subset of training points (support vectors) in the decision function.
    - Many different Kernel functions can be specified for the decision function.
        - Linear
        - Polynomial
        - RBF
        - Sigmoid
        - Custom

The disadvantages of support vector machines include:
    - Beware of overfitting when num_features > num_samples.
    - Choice of Kernel and Regularization can have a large impact on performance
    - No probability estimates
'''