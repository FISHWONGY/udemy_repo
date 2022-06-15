# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

"""
# Bagging Machine Learning Algorithm
"""

df = sns.load_dataset('titanic')

df.shape

df.head()
df.dropna(inplace=True)
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

X['sex'] = lb.fit_transform(X['sex'])

# Data Exploration
X.head()
X.shape
X.describe()
X.info()

y = subset['survived']
y.value_counts()


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


"""***
# Boosting (Hypothesis Boosting)
* Combine several weak learners into a strong learner. 
* Train predictors sequentially

# AdaBoost / Adaptive Boosting
[Robert Schapire](http://rob.schapire.net/papers/explaining-adaboost.pdf)
[Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
[Chris McCormick](http://mccormickml.com/2013/12/13/adaboost-tutorial/)
[Scikit Learn AdaBoost](http://scikit-learn.org/stable/modules/ensemble.html#adaboost)
1995

As above for Boosting:
* Similar to human learning, the algo **learns from past mistakes by focusing more on difficult problems** it did not 
get right in prior learning. 
* In machine learning speak, it pays more attention to training instances that previously underfitted.

Source: Scikit-Learn:

* Fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small 
decision trees) on repeatedly modified versions of the data. 
* The predictions from all of them are then combined through a weighted majority vote (or sum) to 
produce the final prediction.

* The data modifications at each so-called boosting iteration consist of applying weights $w_1, w_2, …, w_N$ 
to each of the training samples. 

* Initially, those weights are all set to $w_i = 1/N$, so that the first step simply 
trains a weak learner on the original data. 
* For each successive iteration, the sample weights are individually modified and the learning algorithm is 
reapplied to the reweighted data. 
* At a given step, those training examples that were incorrectly predicted by the boosted model induced at the 
previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly. 
* As iterations proceed, examples that are difficult to predict receive ever-increasing influence. 
Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.
"""

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)

ada_clf.fit(X_train, y_train)

"""
[SAMME16](https://web.stanford.edu/~hastie/Papers/samme.pdf) 
(Stagewise Additive Modeling using a Multiclass Exponential loss function).
R stands for real
"""

print_score(ada_clf, X_train, X_test, y_train, y_test, train=True)
print("\n*****************************\n")
print_score(ada_clf, X_train, X_test, y_train, y_test, train=False)


"""## AdaBoost with Random Forest"""
from sklearn.ensemble import RandomForestClassifier

ada_clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100), n_estimators=100)

ada_clf.fit(X_train, y_train)

print_score(ada_clf, X_train, X_test, y_train, y_test, train=True)
print("\n*****************************\n")
print_score(ada_clf, X_train, X_test, y_train, y_test, train=False)
# Perform better than just adaboost

# Try using Grid Search
from sklearn.model_selection import GridSearchCV

ada_clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=100), n_estimators=100)

params_grid = {'base_estimator__max_depth': [i for i in range(2, 11, 2)],
               'base_estimator__min_samples_leaf': [5, 10],
               'learning_rate': [0.01, 0.1]}

grid_search = GridSearchCV(ada_clf, params_grid,
                           n_jobs=-1, cv=2,
                           verbose=1, scoring='accuracy')

grid_search.fit(X_train, y_train)

print(grid_search.best_score_)

print(grid_search.best_estimator_.get_params())

print_score(grid_search, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(grid_search, X_train, X_test, y_train, y_test, train=False)

"""***
# Gradient Boosting / Gradient Boosting Machine (GBM)
Works for both regression and classification
[Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting)

* Sequentially adding predictors
* Each one correcting its predecessor
* Fit new predictor to the residual errors

Compare this to AdaBoost: 
* Alter instance weights at every iteration

**Step 1. **

  $$Y = F(x) + \epsilon$$

**Step 2. **

  $$\epsilon = G(x) + \epsilon_2$$

  Substituting (2) into (1), we get:
  
  $$Y = F(x) + G(x) + \epsilon_2$$
    
**Step 3. **

  $$\epsilon_2 = H(x)  + \epsilon_3$$

Now:
  
  $$Y = F(x) + G(x) + H(x)  + \epsilon_3$$
  
Finally, by adding weighting  
  
  $$Y = \alpha F(x) + \beta G(x) + \gamma H(x)  + \epsilon_4$$

Gradient boosting involves three elements:

* **Loss function to be optimized**: Loss function depends on the type of problem being solved. In the case of regression problems, mean squared error is used, and in classification problems, logarithmic loss will be used. In boosting, at each stage, unexplained loss from prior iterations will be optimized rather than starting from scratch.

* **Weak learner to make predictions**: Decision trees are used as a weak learner in gradient boosting.

* **Additive model to add weak learners to minimize the loss function**: Trees are added one at a time and existing trees in the model are not changed. The gradient descent procedure is used to minimize the loss when adding trees.
"""

from sklearn.ensemble import GradientBoostingClassifier

gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train)

print_score(gbc_clf, X_train, X_test, y_train, y_test, train=True)
print("\n*****************************\n")
print_score(gbc_clf, X_train, X_test, y_train, y_test, train=False)

"""***

| Classifier | Decision Tree | Bagging | Random Forest | Optimised RF | Extra-Trees | AdaBoost (CART) | AdaBoost (RF) | Gradient Boosting |
|:-|:-|:- |:- |:- |:- |:-|:-| :- |
| Train accuracy score | 0.9528 | 0.9528 | 0.9325 | 0.9264 | 0.9448 | 0.8661 | 0.9528 | 0.9449 |
| Average accuracy score | 0.7724 | 0.7879 | 0.7801 | 0.7059 | 0.7548 | 0.7793 | 0.7353 | 0.7906 |
| SD | 0.1018 | 0.1008 | 0.1474 | 0.1308 | 0.1406 | 0.1172 | 0.0881 | 0.0912 |
| Test accuracy score | 0.7636 | 0.7455 | 0.7895 | 0.6316 | 0.7895 | 0.6545 | 0.7818 | 0.7818 |

***
"""