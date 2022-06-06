# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
# Ensemble of ensembles - model stacking

* **Ensemble with different types of classifiers**: 
  * Different types of classifiers (E.g., logistic regression, decision trees, random forest, etc.) are fitted on the same training data
  * Results are combined based on either 
    * majority voting (classification) or 
    * average (regression)
  

* **Ensemble with a single type of classifier**: 
  * Bootstrap samples are drawn from training data 
  * With each bootstrap sample, model (E.g., Individual model may be decision trees, random forest, etc.) will be fitted 
  * All the results are combined to create an ensemble. 
  * Suitabe for highly flexible models that is prone to overfitting / high variance.

***

## Combining Method

* **Majority voting or average**: 
  * Classification: Largest number of votes (mode) 
  * Regression problems: Average (mean).
  
  
* **Method of application of meta-classifiers on outcomes**: 
  * Binary outcomes: 0 / 1 from individual classifiers
  * Meta-classifier is applied on top of the individual classifiers. 
  
  
* **Method of application of meta-classifiers on probabilities**: 
  * Probabilities are obtained from individual classifiers. 
  * Applying meta-classifier
"""


df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
num_col = list(df.describe().columns)
col_categorical = list(set(df.columns).difference(num_col))
remove_list = ['EmployeeCount', 'EmployeeNumber', 'StandardHours']
col_numerical = [e for e in num_col if e not in remove_list]
attrition_to_num = {'Yes': 0,
                    'No': 1}
df['Attrition_num'] = df['Attrition'].map(attrition_to_num)
col_categorical.remove('Attrition')
df_cat = pd.get_dummies(df[col_categorical])
X = pd.concat([df[col_numerical], df_cat], axis=1)
y = df['Attrition_num']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2)


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
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_train), 
                                                        lb.transform(clf.predict(X_train)))))

        # cv_res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        # print("Average Accuracy: \t {0:.4f}".format(np.mean(cv_res)))
        # print("Accuracy SD: \t\t {0:.4f}".format(np.std(cv_res)))
        
    elif train==False:
        '''
        test performance
        '''
        res_test = clf.predict(X_test)
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))      
        print("ROC AUC: {0:.4f}\n".format(roc_auc_score(lb.transform(y_test), lb.transform(res_test))))


"""## Model 1: Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, X_test, y_train, y_test, train=True)
print_score(tree_clf, X_train, X_test, y_train, y_test, train=False)


"""## Model 2: Random Forest"""

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train.ravel())

print_score(rf_clf, X_train, X_test, y_train, y_test, train=True)
print_score(rf_clf, X_train, X_test, y_train, y_test, train=False)

en_en = pd.DataFrame()

tree_clf.predict_proba(X_train)

en_en['tree_clf'] = pd.DataFrame(tree_clf.predict_proba(X_train))[1]
en_en['rf_clf'] = pd.DataFrame(rf_clf.predict_proba(X_train))[1]
col_name = en_en.columns
en_en = pd.concat([en_en, pd.DataFrame(y_train).reset_index(drop=True)], axis=1)

en_en.head()

tmp = list(col_name)
tmp.append('ind')
en_en.columns = tmp

en_en.head()


"""# Meta Classifier"""

from sklearn.linear_model import LogisticRegression

m_clf = LogisticRegression(fit_intercept=False, solver='lbfgs')

m_clf.fit(en_en[['tree_clf', 'rf_clf']], en_en['ind'])

en_test = pd.DataFrame()

en_test['tree_clf'] = pd.DataFrame(tree_clf.predict_proba(X_test))[1]
en_test['rf_clf'] = pd.DataFrame(rf_clf.predict_proba(X_test))[1]
col_name = en_en.columns
en_test['combined'] = m_clf.predict(en_test[['tree_clf', 'rf_clf']])

col_name = en_test.columns
tmp = list(col_name)
tmp.append('ind')

tmp

en_test = pd.concat([en_test, pd.DataFrame(y_test).reset_index(drop=True)], axis=1)

en_test.columns = tmp

print(pd.crosstab(en_test['ind'], en_test['combined']))

print(round(accuracy_score(en_test['ind'], en_test['combined']), 4))

print(classification_report(en_test['ind'], en_test['combined']))


"""
# Single Classifier
"""

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

df.head()

df.Attrition.value_counts() / df.Attrition.count()

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

class_weight = {0: 0.839, 1: 0.161}

pd.Series(list(y_train)).value_counts() / pd.Series(list(y_train)).count()

forest = RandomForestClassifier(class_weight=class_weight, n_estimators=100)

ada = AdaBoostClassifier(base_estimator=forest, n_estimators=100,
                         learning_rate=0.5, random_state=42)

ada.fit(X_train, y_train.ravel())

print_score(ada, X_train, X_test, y_train, y_test, train=True)
print_score(ada, X_train, X_test, y_train, y_test, train=False)

bag_clf = BaggingClassifier(base_estimator=ada, n_estimators=50,
                            max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=-1,
                            random_state=42)

bag_clf.fit(X_train, y_train.ravel())

print_score(bag_clf, X_train, X_test, y_train, y_test, train=True)
print_score(bag_clf, X_train, X_test, y_train, y_test, train=False)

