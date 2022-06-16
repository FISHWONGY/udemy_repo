import numpy as np
import pandas as pd

"""
# Project Cancer Detection
# Breast Cancer Wisconsin (Diagnostic) Data Set
[Source: UCI](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
[Data Set info](http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names)
"""

col = ['id', 'Clump Thickness', 'Uniformity of Cell Size',
       'Uniformity of Cell Shape', 'Marginal Adhesion',
       'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
       'Normal Nucleoli', 'Mitoses', 'Class']

# !wget https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data

df = pd.read_csv("/Users/yuawong/Documents/GitHub/udemy_repo/The_Complete_Machine_Learning_Course_with_Python/"
                 "data/breast-cancer-wisconsin.data", names=col, header=None)
df.head()

"""# Data Pre-processing"""

np.where(df.isnull())
df.info()
df['Bare Nuclei'].describe()
df['Bare Nuclei'].value_counts()

"""How do we drop the `?`"""
df[df['Bare Nuclei'] == "?"]
df['Class'].value_counts()
df['Bare Nuclei'].replace("?", np.NAN, inplace=True)
df = df.dropna()

"""
Note that for class: 2 is benign, 4 is for malignant
$$\frac{\text{df["Class"]}}{2} - 1$$
"""

df['Bare Nuclei'].value_counts()
df['Class'] = df['Class'] / 2 - 1
df['Class'].value_counts()
df.columns
df.info()

X = df.drop(['id', 'Class'], axis=1)
X_col = X.columns

y = df['Class']


from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(X.values)


"""Training"""
from sklearn.model_selection import train_test_split

df1 = pd.DataFrame(X, columns=X_col)
df1.head()

X_train, X_test, y_train, y_test = train_test_split(df1, y,
                                                    train_size=0.8,
                                                    random_state=42)

from sklearn.preprocessing import MinMaxScaler
pd.DataFrame(MinMaxScaler().fit_transform(df.drop(['id', 'Class'], axis=1).values), columns=X_col).head()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,
                           p=2, metric='minkowski')

knn.fit(X_train, y_train)

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


print_score(knn, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(knn, X_train, X_test, y_train, y_test, train=False)
# Test Result: accuracy score: 0.9562; ROC AUC: 0.9506

"""# Grid Search"""
from sklearn.model_selection import GridSearchCV

knn.get_params()

params = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

grid_search_cv = GridSearchCV(KNeighborsClassifier(),
                              params, 
                              n_jobs=-1,
                              verbose=1,
                              cv=10)

grid_search_cv.fit(X_train, y_train)

print(grid_search_cv.best_estimator_)

print_score(grid_search_cv, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(grid_search_cv, X_train, X_test, y_train, y_test, train=False)
# Test Result: accuracy score: 0.9562; ROC AUC: 0.9506

print(grid_search_cv.best_params_)
print(grid_search_cv.cv_results_['mean_test_score'])
print(grid_search_cv.cv_results_)


# Test with other model
"""SVM, Random Forest, XGBoost"""
from sklearn import svm
clf = svm.SVC(kernel='rbf', gamma="scale")
clf.fit(X_train, y_train)
print_score(clf, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(clf, X_train, X_test, y_train, y_test, train=False)
# Test Result: accuracy score: 0.9635; ROC AUC: 0.9615


''''# Random Forest'''
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=42, n_estimators=100)
clf.fit(X_train, y_train)
print_score(clf, X_train, X_test, y_train, y_test, train=True)
print("\n******************************\n")
print_score(clf, X_train, X_test, y_train, y_test, train=False)
# Test Result: accuracy score: 0.9489; ROC AUC: 0.9419

# So SVM > KNN > Random Forest

