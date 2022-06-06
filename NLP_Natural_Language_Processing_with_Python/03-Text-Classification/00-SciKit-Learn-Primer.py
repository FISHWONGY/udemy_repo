# # Scikit-learn Primer
import numpy as np
import pandas as pd

df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t')
df.head()

len(df)


# ## Check for missing values:
df.isnull().sum()


# ## Take a quick look at the *ham* and *spam* `label` column:
df['label'].unique()

df['label'].value_counts()


# We see that 4825 out of 5572 messages, or 86.6%, are ham.<br>This means that any machine learning model we create has to perform **better than 86.6%** to beat random chance.</font>

# ## Visualize the data:
# Since we're not ready to do anything with the message text, let's see if we can predict ham/spam labels based on message length and punctuation counts. We'll look at message `length` first:
df['length'].describe()


# This dataset is extremely skewed. The mean value is 80.5 and yet the max length is 910. Let's plot this on a logarithmic x-axis.</font>

from IPython import get_ipython
import matplotlib.pyplot as plt
get_ipython().run_line_magic('inline')

plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(df[df['label']=='ham']['length'],bins=bins,alpha=0.8)
plt.hist(df[df['label']=='spam']['length'],bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()


# It looks like there's a small range of values where a message is more likely to be spam than ham.</font>
# 
# Now let's look at the `punct` column:
df['punct'].describe()


plt.xscale('log')
bins = 1.5**(np.arange(0, 15))
plt.hist(df[df['label']=='ham']['punct'],bins=bins,alpha=0.8)
plt.hist(df[df['label']=='spam']['punct'],bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()


# This looks even worse - there seem to be no values where one would pick spam over ham. We'll still try to build a machine learning classification model, but we should expect poor results.</font>
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
print(metrics.confusion_matrix(y_test,predictions))

# You can make the confusion matrix less confusing by adding labels:
df = pd.DataFrame(metrics.confusion_matrix(y_test,predictions), index=['ham','spam'], columns=['ham','spam'])
df


# These results are terrible! More spam messages were confused as ham (241) than correctly identified as spam (5), although a relatively small number of ham messages (46) were confused as spam.</font>

# Print a classification report
print(metrics.classification_report(y_test, predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


#This model performed *worse* than a classifier that assigned all messages as "ham" would have!</font>

# ___
# # Train a naïve Bayes classifier:
# One of the most common - and successful - classifiers is [naïve Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes).

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)


# ## Run predictions and report on metrics
predictions = nb_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))


# The total number of confusions dropped from **287** to **256**. [241+46=287, 246+10=256]

print(metrics.classification_report(y_test,predictions))


print(metrics.accuracy_score(y_test,predictions))


# Better, but still less accurate than 86.6%

# ___
# # Train a support vector machine (SVM) classifier
# Among the SVM options available, we'll use [C-Support Vector Classification (SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC)

from sklearn.svm import SVC
svc_model = SVC(gamma='auto')
svc_model.fit(X_train,y_train)


# ## Run predictions and report on metrics

predictions = svc_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))


# The total number of confusions dropped even further to **209**.

print(metrics.classification_report(y_test,predictions))


print(metrics.accuracy_score(y_test,predictions))


# And finally we have a model that performs *slightly* better than random chance

# Great! Now you should be able to load a dataset, divide it into training and testing sets, and perform simple analyses using scikit-learn.
# ## Next up: Feature Extraction from Text
