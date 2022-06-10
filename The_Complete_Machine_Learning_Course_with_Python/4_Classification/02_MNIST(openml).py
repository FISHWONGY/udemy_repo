import numpy as np
import sklearn
from sklearn.datasets import fetch_openml
mnist = fetch_openml(name='mnist_784')
mnist

len(mnist['data'])

# Visualisation
X, y = mnist['data'], mnist['target']
X
X.shape
28*28
y

y = y.astype("float")

# Instead of DF, X needs to be array
X = X.to_numpy()
X[69999]
y[69999]
y.shape

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
def viz(n):
    plt.imshow(X[n].reshape(28, 28))
    return


viz(69999)
y[1000]
viz(1000)


# Exercise: Locating the number 4 and plot the image
type(y)
y == 4
np.where(y == 4)
# array([    2,     9,    20, ..., 69977, 69987, 69997]) All of these are 4's hand writing

y[69977]

_ = X[69977]
_image = _.reshape(28, 28)
plt.imshow(_image)

viz(69977)


# Splitting the train and test sets
# Method #1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Methond #2
num_split = 60000
X_train, X_test, y_train, y_test = X[:num_split], X[num_split:], y[:num_split], y[num_split:]

# Shuffling the dataset
shuffle_index = np.random.permutation(num_split)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

'''
# Training a Binary Classifier
What the code below doing is split the data into 2 classes
1 class - 'zero'
2 class - 'non-zero'
'''
y_train_0 = (y_train == 0)
y_train_0

y_test_0 = (y_test == 0)
y_test_0

# At this point we can pick any classifier and train it.
# This is the iterative part of choosing and testing all the classifiers and tuning the hyper parameters


# SGDClassifier
# Training
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(random_state=0)
clf.fit(X_train, y_train_0)

#  Prediction
viz(1000) # 0

clf.predict(X[1000].reshape(1, -1)) # true = predicted correctly it is 0

viz(1001) # 4

clf.predict(X[1001].reshape(1, -1)) # false = predicted correctly it is not 0

'''
###
# Performance Measures
'''
# Measuring Accuracy Using Cross-Validation
# StratifiedKFold
'''
StratifiedKFold utilised the Stratified sampling concept
 - The population is divided into homogeneous subgroups called strata
 - The right number of instances is sampled from each stratum
 - To guarantee that the test set is representative of the population
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
clf = SGDClassifier(random_state=0)

skfolds = StratifiedKFold(n_splits=3, random_state=100, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_0):
    clone_clf = clone(clf)
    X_train_fold = X_train[train_index]
    y_train_folds = (y_train_0[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_0[test_index])

    clone_clf.fit(X_train_fold, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print("{0:.4f}".format(n_correct / len(y_pred)))


# cross_val_score using K-fold Cross-Validation
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X_train, y_train_0, cv=3, scoring='accuracy')


# Exercise:
# What if you would like to perform 10-fold CV test? How would you do that
cross_val_score(clf, X_train, y_train_0, cv=10, scoring='accuracy')

# Danger of Blindly Applying Evaluator As a Performance Measure
1 - sum(y_train_0) / len(y_train_0)
'''
A simple check shows that 90.1% of the images are not zero. Any time you guess the image is not zero, 
you will be right 90.13% of the time.
Bare this in mind when you are dealing with skewed datasets. Because of this, accuracy is generally 
not the preferred performance measure for classifiers.
'''

# Confusion Matrix
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(clf, X_train, y_train_0, cv=3)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_0, y_train_pred)

import pandas as pd
pd.DataFrame(confusion_matrix(y_train_0, y_train_pred))

pd.DataFrame(confusion_matrix(y_train_0, y_train_pred),
             columns=pd.MultiIndex.from_product([['Prediction'], ["Negative", "Positive"]]),
             index=pd.MultiIndex.from_product([["Actual"], ["Negative", "Positive"]]))


# Precision
'''
precision = True Positives / True Positives + False Positives
'''
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_0, y_train_pred) # 5618 / (574 + 5618)
5618 / (574 + 5618)

# Recall
'''
recall = True Positives / True Positives + False Negatives
'''
recall_score(y_train_0, y_train_pred) # 5618 / (305 + 5618)

5618 / (305 + 5618)


# F1 Score
# 𝐹1 score is the harmonic mean of precision and recall. Regular mean gives equal weight to all values.
# Harmonic mean gives more weight to low values.
from sklearn.metrics import f1_score
f1_score(y_train_0, y_train_pred)


# Precision / Recall Tradeoff
# Increasing precision reduced recall and vice versa
clf = SGDClassifier(random_state=0)
clf.fit(X_train, y_train_0)

y[1000]

y_scores = clf.decision_function(X[1000].reshape(1, -1))
y_scores

threshold = 0

y_some_digits_pred = (y_scores > threshold)

y_some_digits_pred

threshold = 40000
y_some_digits_pred = (y_scores > threshold)
y_some_digits_pred

y_scores = cross_val_predict(clf, X_train, y_train_0, cv=3, method='decision_function')

plt.figure(figsize=(12, 8))
plt.hist(y_scores, bins=100)

# With the decision scores, we can compute precision and recall for all possible thresholds using the precision_recall_curve() function:
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_0, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([-0.5, 1.5])


plt.figure(figsize=(12, 8))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

'''
With this chart, you can select the threshold value that gives you the best precision/recall tradeoff for your task.

Some tasks may call for higher precision (accuracy of positive predictions). Like designing a classifier that picks up 
adult contents to protect kids. This will require the classifier to set a high bar to allow any contents to be consumed by children.

Some tasks may call for higher recall (ratio of positive instances that are correctly detected by the classifier). 
Such as detecting shoplifters/intruders on surveillance images - Anything that remotely resemble "positive" instances to be picked up.
'''

# One can also plot precisions against recalls to assist with the threshold selection
plt.figure(figsize=(12, 8))
plt.plot(precisions, recalls)
plt.xlabel('recalls')
plt.ylabel('precisions')
plt.title('PR Curve: precisions/recalls tradeoff')


'''
# Setting High Precisions
'''
# Let's aim for 90% precisions.
len(precisions)
len(thresholds)
plt.figure(figsize=(12, 8))
plt.plot(thresholds, precisions[1:])

idx = len(precisions[precisions < 0.9])
thresholds[idx]

y_train_pred_90 = (y_scores > thresholds[idx])

precision_score(y_train_0, y_train_pred_90)
recall_score(y_train_0, y_train_pred_90)


# Let's aim for 99% precisions.
idx = len(precisions[precisions < 0.99])
thresholds[idx]
y_train_pred_90 = (y_scores > thresholds[idx])
precision_score(y_train_0, y_train_pred_90)
recall_score(y_train_0, y_train_pred_90)


'''
Setting High Recall
'''
# Exercise
# High Recall Score. Recall score > 0.9
idx = len(recalls[recalls > 0.9])
thresholds[idx]

y_train_pred_90 = (y_scores > thresholds[idx])
precision_score(y_train_0, y_train_pred_90)
recall_score(y_train_0, y_train_pred_90)


# The Receiver Operating Characteristics (ROC) Curve
'''
Instead of plotting precision versus recall, the ROC curve plots the true positive rate (another name for recall) 
against the false positive rate. The false positive rate (FPR) is the ratio of negative instances that are incorrectly 
classified as positive. 
It is equal to one minus the true negative rate, 
which is the ratio of negative instances that are correctly classified as negative.

The TNR is also called specificity. 
Hence the ROC curve plots sensitivity (recall) versus 1 - specificity.
'''
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_0, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')


plt.figure(figsize=(10, 8))
plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_0, y_scores)

'''
Use PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives
Use ROC curve whenever the negative class is rare or when you care more about the false negatives than the false positives
'''

# Model Comparison
# Random Forest
from sklearn.ensemble import RandomForestClassifier
f_clf = RandomForestClassifier(random_state=0, n_estimators=100)
y_probas_forest = cross_val_predict(f_clf, X_train, y_train_0,
                                    cv=3, method='predict_proba')

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_0, y_scores_forest)

plt.figure(figsize=(12, 8))
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_0, y_scores_forest)

f_clf.fit(X_train, y_train_0)
y_train_rf = cross_val_predict(f_clf, X_train, y_train_0, cv=3)

precision_score(y_train_0, y_train_rf)
recall_score(y_train_0, y_train_rf)

confusion_matrix(y_train_0, y_train_rf)

