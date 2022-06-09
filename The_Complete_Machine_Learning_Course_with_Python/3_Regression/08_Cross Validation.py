import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

boston = datasets.load_boston()

# Cross Validation (CV)
'''
- Hold out Cross Validation
- k-fold Cross Validation

A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV.

In the basic approach, called k-fold CV, the training set is split into k smaller sets. 
The following procedure is followed for each of the k “folds”:
- A model is trained using k-1 of the folds as training data;
- the resulting model is validated on the remaining part of the data (i.e., it is used as a test set to compute a 
performance measure such as accuracy).

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
'''

# Holdout Method
'''
- Split initial dataset into a separate training and test dataset
- Training dataset - model training
- Test dataset - estimate its generalisation performance

A variation is to split the training set to two :- training set and validation set

Training set:- For fitting different models
Validation set :- For tuning and comparing different parameter settings to further improve the 
performance for making predictions on unseen data. And finally for model selection.

This process is called model selection. We want to select the optimal values of tuning parameters (also called hyperparameters).
'''

# K-fold Cross-validation
'''
- Randomly split the training dataset into k folds without replacement.
- k — 1 folds are used for the model training.
- The one fold is used for performance evaluation.

This procedure is repeated k times.

Final outcomes:- k models and performance estimates.

- calculate the average performance of the models based on the different, 
independent folds to obtain a performance estimate that is less sensitive to the sub-partitioning of the training data 
compared to the holdout method.

- k-fold cross-validation is used for model tuning. Finding the optimal hyperparameter values that yields a satisfying 
generalization performance.

- Once we have found satisfactory hyperparameter values, we can retrain the model on the complete training set and 
obtain a final performance estimate using the independent test set. The rationale behind fitting a model to the whole 
training dataset after k-fold cross-validation is that providing more training samples to a learning algorithm usually 
results in a more accurate and robust model.

- Common k is 10

- For relatively small training sets, increase the number of folds.
'''

# Stratified k-fold cross-validation
'''
- variation of k-fold
- Can yield better bias and variance estimates, especially in cases of unequal class proportions
'''


# Cross-validation: evaluating estimator performance
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.4, random_state=0)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

regression = svm.SVR(kernel='linear', C=1).fit(X_train, y_train)
regression.score(X_test, y_test)

'''
When evaluating different settings (“hyperparameters”) for estimators, such as the C setting that must be manually set 
for an SVM, there is still a risk of overfitting on the test set because the parameters can be tweaked until 
the estimator performs optimally.

This way, knowledge about the test set can “leak” into the model and evaluation metrics no longer report on generalization performance.

To solve this problem, yet another part of the dataset can be held out as a so-called “validation set”: 
training proceeds on the training set, after which evaluation is done on the validation set, 
and when the experiment seems to be successful, final evaluation can be done on the test set.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can be 
used for learning the model, and the results can depend on a particular random choice for the pair of (train, validation) sets.

A solution to this problem, as discussed earlier, is a procedure called cross-validation (CV for short). 
A test set should still be held out for final evaluation, but the validation set is no longer needed when doing CV. 
In the basic approach, called k-fold CV, the training set is split into k smaller sets 
(other approaches are described below, but generally follow the same principles). 
The following procedure is followed for each of the k “folds”:

- A model is trained using k-1 of the folds as training data;
- the resulting model is validated on the remaining part of the data 
 (i.e., it is used as a test set to compute a performance measure such as accuracy).

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop. 
This approach can be computationally expensive, but does not waste too much data 
(as it is the case when fixing an arbitrary test set), 
which is a major advantage in problem such as inverse inference where the number of samples is very small.
'''

# Computing cross-validated metrics
from sklearn.model_selection import cross_val_score
regression = svm.SVR(kernel='linear', C=1)
scores = cross_val_score(regression, boston.data, boston.target, cv=5)
scores

# The mean score and the 95% confidence interval of the score estimate are hence given by:
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

# By default, the score computed at each CV iteration is the score method of the estimator.
# It is possible to change this by using the scoring parameter:

scores = cross_val_score(regression, boston.data, boston.target, cv=5, scoring='neg_mean_squared_error')
scores


# K-fold
'''
KFold divides all the samples in k groups of samples, called folds 
(if k = n, this is equivalent to the Leave One Out strategy), of equal sizes (if possible). 
The prediction function is learned using k - 1 folds, and the fold left out is used for test.

Example of 2-fold cross-validation on a dataset with 4 samples:
'''

from sklearn.model_selection import KFold

X = ["a", "b", "c", "d"]
kf = KFold(n_splits=2)
for train, test in kf.split(X):
    print("%s %s" % (train, test))


# Stratified k-fold
'''
StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately 
the same percentage of samples of each target class as the complete set.

Example of stratified 3-fold cross-validation on a dataset with 10 samples from two slightly unbalanced classes
'''
from sklearn.model_selection import StratifiedKFold

X = np.ones(10)
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print("%s %s" % (train, test))

print(X)
print(y)


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
pipe_svm = make_pipeline(StandardScaler(),
                         PCA(n_components=2),
                         svm.SVR(kernel='linear', C=1)) # LogisticRegression(random_state=1)
pipe_svm.fit(X_train, y_train)
y_pred = pipe_svm.predict(X_test)
print('Test Accuracy: %.3f' % pipe_svm.score(X_test, y_test))


scores = cross_val_score(estimator=pipe_svm,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)

print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))











