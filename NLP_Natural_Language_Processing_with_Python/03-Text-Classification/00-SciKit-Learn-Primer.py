# # Scikit-learn Primer
import numpy as np
import pandas as pd

df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/smsspamcollection.tsv', sep='\t')
print(df.head())

print(len(df))


# ## Check for missing values:
print(df.isnull().sum())


# ## Take a quick look at the *ham* and *spam* `label` column:
print(df['label'].unique())

print(df['label'].value_counts())


# We see that 4825 out of 5572 messages, or 86.6%, are ham.<br>This means that any machine learning model we create has to perform **better than 86.6%** to beat random chance.</font>

# ## Visualize the data:
# Since we're not ready to do anything with the message text, let's see if we can predict ham/spam labels based on message length and punctuation counts. We'll look at message `length` first:
print(df['length'].describe())


# This dataset is extremely skewed. The mean value is 80.5 and yet the max length is 910. Let's plot this on a logarithmic x-axis.</font>
import matplotlib.pyplot as plt


plt.xscale('log')
bins = 1.15**(np.arange(0, 50))
plt.hist(df[df['label'] == 'ham']['length'], bins=bins, alpha=0.8)
plt.hist(df[df['label'] == 'spam']['length'], bins=bins, alpha=0.8)
plt.legend(('ham', 'spam'))
plt.show()
# It looks like there's a small range of values where a message is more likely to be spam than ham.</font>

#
# Now let's look at the `punct` column:
print(df['punct'].describe())


plt.xscale('log')
bins = 1.5**(np.arange(0, 15))
plt.hist(df[df['label'] == 'ham']['punct'], bins=bins, alpha=0.8)
plt.hist(df[df['label'] == 'spam']['punct'], bins=bins, alpha=0.8)
plt.legend(('ham', 'spam'))
plt.show()
# This looks even worse - there seem to be no values where one would pick spam over ham.
# We'll still try to build a machine learning classification model, but we should expect poor results.</font>

# Split the data into train & test sets:
# If we wanted to divide the DataFrame into two smaller sets, we could use
# > `train, test = train_test_split(df)`
# For our purposes let's also set up our Features (X) and Labels (y). The Label is simple - we're trying to predict the `label` column in our data. For Features we'll use the `length` and `punct` columns. *By convention, **X** is capitalized and **y** is lowercase.*

# ## Selecting features
# There are two ways to build a feature set from the columns we want. If the number of features is small, then we can pass those in directly:
# > `X = df[['length','punct']]`
# 
# If the number of features is large, then it may be easier to drop the Label and any other unwanted columns:
# > `X = df.drop(['label','message'], axis=1)`
# 
# These operations make copies of **df**, but do not change the original DataFrame in place. All the original data is preserved.

# Create Feature and Label sets
# X - df; y - Series
X = df[['length', 'punct']]  # note the double set of brackets
y = df['label']


# ## Additional train/test/split arguments:
# The default test size for `train_test_split` is 30%. Here we'll assign 33% of the data for testing.<br>
# Also, we can set a `random_state` seed value to ensure that everyone uses the same "random" training & testing sets.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print('Training Data Shape:', X_train.shape)
print('Testing Data Shape: ', X_test.shape)


# Now we can pass these sets into a series of different training & testing algorithms and compare their results.
# One of the simplest multi-class classification tools is [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). Scikit-learn offers a variety of algorithmic solvers; we'll use [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS).

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='lbfgs')

lr_model.fit(X_train, y_train)


# ## Test the Accuracy of the Model
from sklearn import metrics

# Create a prediction set:
predictions = lr_model.predict(X_test)

# Print a confusion matrix
print(metrics.confusion_matrix(y_test, predictions))

# You can make the confusion matrix less confusing by adding labels:
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham', 'spam'],
                  columns=['ham', 'spam'])
print(df.head())


# These results are terrible! More spam messages were confused as ham (241) than correctly identified as spam (5), although a relatively small number of ham messages (46) were confused as spam.</font>

# Print a classification report
print(metrics.classification_report(y_test, predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test, predictions))
# This model performed *worse* than a classifier that assigned all messages as "ham" would have!</font>

# ___
# # Train a naïve Bayes classifier:
# One of the most common - and successful - classifiers is [naïve Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes).

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)


# ## Run predictions and report on metrics
predictions = nb_model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))


# The total number of confusions dropped from **287** to **256**. [241+46=287, 246+10=256]

print(metrics.classification_report(y_test, predictions))


print(metrics.accuracy_score(y_test, predictions))


# Better, but still less accurate than 86.6%

# ___
# # Train a support vector machine (SVM) classifier
# Among the SVM options available, we'll use [C-Support Vector Classification (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

from sklearn.svm import SVC
svc_model = SVC(gamma='auto')
svc_model.fit(X_train, y_train)


# ## Run predictions and report on metrics

predictions = svc_model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))


# The total number of confusions dropped even further to **209**.

print(metrics.classification_report(y_test, predictions))

print(metrics.accuracy_score(y_test, predictions))
# 0.8863
# And finally we have a model that performs *slightly* better than random chance

# Great! Now you should be able to load a dataset, divide it into training and testing sets,
# and perform simple analyses using scikit-learn.
# ## Next up: Feature Extraction from Text

# TESTING MY OWN STUFF
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

    elif train == False:
        '''
        test performance
        '''
        print("Test Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))


df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/smsspamcollection.tsv', sep='\t')
label_dict = {"ham": 0, "spam": 1}
df = df.replace({"label": label_dict})

X = df[['length', 'punct']]  # note the double set of brackets
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3,
                            n_jobs=-1)

xgb_clf.fit(X_train, y_train)

print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)
print("\n******************************\n")
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)
# accuracy score: 0.8787


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# Create the parameter grid
param_grid = {'learning_rate': np.linspace(0.1, 2, 150),
              'min_samples_leaf': list(range(20, 65))}

# Create a random search object
random_GBM_class = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(),
    param_distributions=param_grid,
    n_iter=10,
    scoring='accuracy', n_jobs=4, cv=5, refit=True, return_train_score=True
)

# Fit to the training data
random_GBM_class.fit(X_train, y_train)

# Print the values used for both hyperparameters
print(random_GBM_class.cv_results_['param_learning_rate'])
print(random_GBM_class.cv_results_['param_min_samples_leaf'])

# Get best params
print(random_GBM_class.best_params_)
# {'min_samples_leaf': 42, 'learning_rate': 0.4697986577181208}
print(random_GBM_class.best_score_)
# 0.89205


gbc_model = GradientBoostingClassifier(learning_rate=0.4697986577181208, min_samples_leaf=42)
gbc_model.fit(X_train, y_train)
# ## Run predictions and report on metrics
predictions = gbc_model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
# The total number of confusions dropped even further to **209**.
print(metrics.classification_report(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))
# 0.888526
