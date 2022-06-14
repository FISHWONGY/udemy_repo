# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

"""
# Bagging Machine Learning Algorithm

### Bootstrap Aggregating or [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
* [Scikit- Learn Reference](http://scikit-learn.org/stable/modules/ensemble.html#bagging)
* Bootstrap sampling: Sampling with replacement
* Combine by averaging the output (regression)
* Combine by voting (classification)
* Can be applied to many classifiers which includes ANN, CART, etc.
"""

# Data Exploration
df = sns.load_dataset('titanic')
print(df.shape)
df.head()
df['pclass'].unique()
df['pclass'].value_counts()
df['sex'].unique()
df['sex'].value_counts()
df['age'].hist(bins=50)


"""## Data Pre-processing"""
subset = df[['pclass', 'sex', 'age', 'survived']].copy()
subset.dropna(inplace=True)
X = subset[['pclass', 'sex', 'age']].copy()

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

# Change female, male to 0 & 1
X['sex'] = lb.fit_transform(subset['sex'])

# Data Exploration
X.head()
print(X.shape)
X.describe()
X.info()

y = subset['survived']
y.value_counts()

"""***
# Random Forest
[paper](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf)
* Ensemble of Decision Trees
* Training via the bagging method (Repeated sampling with replacement)
  * Bagging: Sample from samples
  * RF: Sample from predictors. $m=sqrt(p)$ for classification and $m=p/3$ for regression problems.

* Utilise uncorrelated trees

Random Forest
* Sample both observations and features of training data

Bagging
* Samples only observations at random
* Decision Tree select best feature when splitting a node
"""

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score
def print_score(clf, X_train, X_test, y_train, y_test, train=True):
    '''
    v0.1 Follow the scikit learn library format in terms of input
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_train)
    if train:
        '''
        training performance
        '''
        res = clf.predict(X_train)
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, 
                                                                res)))
        print("Classification Report: \n {}\n".format(classification_report(y_train, 
                                                                            res)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, 
                                                                  res)))
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_train), 
                                                      lb.transform(res))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        res_test = clf.predict(X_test)
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, 
                                                                res_test)))
        print("Classification Report: \n {}\n".format(classification_report(y_test, 
                                                                            res_test)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, 
                                                                  res_test)))   
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_test), 
                                                      lb.transform(res_test))))

rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)

rf_clf.fit(X_train, y_train)

print_score(rf_clf, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(rf_clf, X_train, X_test, y_train, y_test, train=False)

"""## Grid Search"""

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

rf_clf = RandomForestClassifier(random_state=42, n_estimators=100)

params_grid = {"max_depth": [3, None],
               "min_samples_split": [2, 3, 10],
               "min_samples_leaf": [1, 3, 10],
               "bootstrap": [True, False],
               "criterion": ['gini', 'entropy']}

grid_search = GridSearchCV(rf_clf, params_grid,
                           n_jobs=-1, cv=5,
                           verbose=1, scoring='accuracy',
                           iid=False)

grid_search.fit(X_train, y_train)

grid_search.best_score_

grid_search.best_estimator_.get_params()

print_score(grid_search, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(grid_search, X_train, X_test, y_train, y_test, train=False)

"""***

# Extra-Trees (Extremely Randomized Trees) Ensemble

[scikit-learn](http://scikit-learn.org/stable/modules/ensemble.html#bagging)

* Random Forest is build upon Decision Tree
* Decision Tree node splitting is based on gini or entropy or some other algorithms
* Extra-Trees make use of random thresholds for each feature unlike Decision Tree
"""

from sklearn.ensemble import ExtraTreesClassifier

xt_clf = ExtraTreesClassifier(random_state=42, n_estimators=100)

xt_clf.fit(X_train, y_train)

print_score(xt_clf, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(xt_clf, X_train, X_test, y_train, y_test, train=False)

"""***"""

def eval_gini(y_true, y_prob):
    """
    Original author CPMP : https://www.kaggle.com/cpmpml
    In kernel : https://www.kaggle.com/cpmpml/extremely-fast-gini-computation
    """
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    ntrue = 0
    gini = 0
    delta = 0
    n = len(y_true)
    for i in range(n-1, -1, -1):
        y_i = y_true[i]
        ntrue += y_i
        gini += y_i * delta
        delta += 1 - y_i
    gini = 1 - 2 * gini / (ntrue * (n - ntrue))
    return gini

