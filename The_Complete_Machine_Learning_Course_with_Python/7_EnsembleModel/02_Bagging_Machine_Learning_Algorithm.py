import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

"""
Bagging Machine Learning Algorithm

### **Bootstrap Aggregating or [Bagging](https://en.wikipedia.org/wiki/Bootstrap_aggregating)
* [Scikit- Learn Reference](http://scikit-learn.org/stable/modules/ensemble.html#bagging)
* Bootstrap sampling: Sampling with replacement
* Combine by averaging the output (regression)
* Combine by voting (classification)
* Can be applied to many classifiers which includes ANN, CART, etc.
"""

df = sns.load_dataset('titanic')

# Data Exploration
print(df.shape)
df.head()
# df.dropna(inplace=True)
df['pclass'].unique()
df['pclass'].value_counts()
df['sex'].unique()
df['sex'].value_counts()
df['age'].hist(bins=50) # Pretty skewed dataset

"""## Data Pre-processing"""
subset = df[['pclass', 'sex', 'age', 'survived']].copy()
subset.dropna(inplace=True)

X = subset[['pclass', 'sex', 'age']].copy()

from sklearn import preprocessing
# lb = preprocessing.LabelBinarizer()
le = preprocessing.LabelEncoder()

# Change sex: female, makle to 0 & 1
X['sex'] = le.fit_transform(subset['sex'])

# Data Exploration
X.head()
print(X.shape)
X.describe()
X.info()

y = subset['survived'].copy()

y.value_counts()

"""Fit Model"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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


"""## Decision Tree"""

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

# Score/ Results for training set
print_score(clf, X_train, X_test, y_train, y_test, train=True)
# Score/ Results for test set
print("\n********************************\n")
print_score(clf, X_train, X_test, y_train, y_test, train=False)

"""***
## Bagging (oob_score=False)
"""

bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=1000,
                            bootstrap=True, n_jobs=-1,
                            random_state=42)

bag_clf.fit(X_train, y_train)

# Score/ Results for training set
print_score(bag_clf, X_train, X_test, y_train, y_test, train=True)
# Score/ Results for test set
print("\n********************************\n")
print_score(bag_clf, X_train, X_test, y_train, y_test, train=False)

"""***

## Bagging (oob_score=True)

Use out-of-bag samples to estimate the generalization accuracy
"""

bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=1000,
                            bootstrap=True, oob_score=True,
                            n_jobs=-1, random_state=42)

bag_clf.fit(X_train, y_train)

print(bag_clf.oob_score_)

# Score/ Results for training set
print_score(bag_clf, X_train, X_test, y_train, y_test, train=True)
# Score/ Results for test set
print("\n********************************\n")
print_score(bag_clf, X_train, X_test, y_train, y_test, train=False)

