# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

"""
# XGBoost (Extreme Gradient Boosting)

[Documentation](http://xgboost.readthedocs.io/en/latest/)

[tqchen github](https://github.com/tqchen/xgboost/tree/master/demo/guide-python)

[dmlc github](https://github.com/dmlc/xgboost)

* “Gradient Boosting” is proposed in the paper Greedy Function Approximation: A Gradient Boosting Machine, by Friedman. 
* XGBoost is based on this original model. 

* Supervised Learning

## Objective Function : Training Loss + Regularization

$$Obj(Θ)=L(θ)+Ω(Θ)$$

* $L$ is the training loss function, and 
* $Ω$ is the regularization term. 

### Training Loss

The training loss measures how predictive our model is on training data.

Example 1, Mean Squared Error for Linear Regression:

$$L(θ)= \sum_i(y_i-\hat{y}_i)^2$$

Example 2, Logistic Loss for Logistic Regression:

$$ L(θ) = \sum_i \large[ y_i ln(1 + e^{-\hat{y}_i}) + (1-y_i) ln(1 + e^{\hat{y}_i}) \large] $$

### Regularization Term

The regularization term controls the complexity of the model, which helps us to avoid overfitting.

[XGBoost vs GBM](https://www.quora.com/What-is-the-difference-between-the-R-gbm-gradient-boosting-machine-and-xgboost-extreme-gradient-boosting/answer/Tianqi-Chen-1)

* Specifically,  xgboost used a more regularized model formalization to control over-fitting, which gives it better performance.

* For model, it might be more suitable to be called as regularized gradient boosting.
"""

df = sns.load_dataset('titanic')

df.dropna(inplace=True)

"""## Data Pre-processing"""

X = df[['pclass', 'sex', 'age']].copy()

from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()

X['sex'] = lb.fit_transform(X['sex'])

y = df['survived']

"""***"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))


"""# XGBoost"""

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3,
                            n_jobs=-1)

xgb_clf.fit(X_train, y_train)

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

"""| Classifier | Decision Tree | Bagging | Random Forest | Optimised RF | Extra-Trees | AdaBoost (CART) | AdaBoost (RF) | Gradient Boosting |
|:-|:-|:- |:- |:- |:- |:-|:-| :- |
| Train accuracy score | 0.9528 | 0.9528 | 0.9325 | 0.9264 | 0.9448 | 0.8661 | 0.9528 | 0.9449 |
| Average accuracy score | 0.7724 | 0.7879 | 0.7801 | 0.7059 | 0.7548 | 0.7793 | 0.7353 | 0.7906 |
| SD | 0.1018 | 0.1008 | 0.1474 | 0.1308 | 0.1406 | 0.1172 | 0.0881 | 0.0912 |
| Test accuracy score | 0.7636 | 0.7455 | 0.7895 | 0.6316 | 0.7895 | 0.6545 | 0.7818 | 0.7818 |

***
"""