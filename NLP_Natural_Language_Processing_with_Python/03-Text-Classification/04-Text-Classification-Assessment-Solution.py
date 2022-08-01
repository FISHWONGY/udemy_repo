import pandas as pd

df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/moviereviews2.tsv', sep='\t')
print(df.head())

# ### Task #2: Check for missing values:
# Check for NaN values:
print(df.isnull().sum())

# Check for whitespace strings (it's OK if there aren't any!):
blanks = []  # start with an empty list

for i, lb, rv in df.itertuples():  # iterate over the DataFrame
    if type(rv) == str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
        
print(len(blanks))


# ### Task #3:  Remove NaN values:
df.dropna(inplace=True)


# ### Task #4: Take a quick look at the `label` column:
print(df['label'].value_counts())


# ### Task #5: Split the data into train & test sets:
# You may use whatever settings you like. To compare your results to the solution notebook, use `test_size=0.33, random_state=42`
from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Task #6: Build a pipeline to vectorize the date, then train and fit a model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()), ])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  


# ### Task #7: Run predictions and analyze the results
# Form a prediction set
predictions = text_clf.predict(X_test)


# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions))

# Print a classification report
print(metrics.classification_report(y_test, predictions))

# Print the overall accuracy
print(metrics.accuracy_score(y_test, predictions))
# 0.9219858

# ## Great job!

import pandas as pd
df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/moviereviews2.tsv', sep='\t')
label_dict = {"neg": 0, "pos": 1}
df = df.replace({"label": label_dict})
df.dropna(inplace=True)
blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)
df.drop(blanks, inplace=True)

stopwords = ['a', 'about', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', \
             'even', 'ever', 'for', 'from', 'get', 'had', 'has', 'have', 'he', 'her', 'hers', 'his', \
             'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'me', 'my', 'of', 'on', 'or', \
             'see', 'seen', 'she', 'so', 'than', 'that', 'the', 'their', 'there', 'they', 'this', \
             'to', 'was', 'we', 'were', 'what', 'when', 'which', 'who', 'will', 'with', 'you']

from sklearn.model_selection import train_test_split
X = df['review']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

model_list = [LinearSVC(), SVC(), MultinomialNB(), GradientBoostingClassifier(), XGBClassifier()]

acc_list = []
for m in model_list:
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', m), ])

    # Feed the training data through the pipeline
    text_clf.fit(X_train, y_train)
    # ## Test the classifier and display results
    predictions = text_clf.predict(X_test)


    # Report the confusion matrix
    from sklearn import metrics
    print('Confusion Metrix for ' + str(m).split('(')[0]
          + ':\n' + str(metrics.confusion_matrix(y_test, predictions)))
    # Print a classification report
    print('classification report for ' + str(m).split('(')[0]
          + ':\n\n' + str(metrics.classification_report(y_test, predictions)))
    # Print the overall accuracy
    print('Accuracy for ' + str(m).split('(')[0]
          + ' is: ' + str(metrics.accuracy_score(y_test, predictions)) + '\n')

    acc_list.append(metrics.accuracy_score(y_test, predictions))


acc_str = ''
for i in range(len(model_list)):
    acc_str += str(model_list[i]).split('(')[0] + ': ' + str(acc_list[i]) + '\n'
print('Accuracy for each model: \n' + acc_str)

acc_dict = dict(zip(model_list, acc_list))
# print("Accuracy for each model: ", sorted(acc_dict, key=acc_dict.get, reverse=True))
print("Model with Highest Accuracy Score: ", max(acc_dict.items(), key=lambda x: x[1]))

"""
TfidfVectorizer(stop_words=stopwords)
Accuracy for each model: 
LinearSVC: 0.914387031408308
SVC: 0.9128672745694022
MultinomialNB: 0.9062816616008106
GradientBoostingClassifier: 0.8495440729483282
XGBClassifier: 0.8738601823708206

Model with Highest Accuracy Score:  (LinearSVC(), 0.914387031408308)
"""

"""
TfidfVectorizer()

Accuracy for each model: 
LinearSVC: 0.9219858156028369
SVC: 0.9184397163120568
MultinomialNB: 0.9052684903748733
GradientBoostingClassifier: 0.8485309017223911
XGBClassifier: 0.8687943262411347
Model with Highest Accuracy Score:  (LinearSVC(), 0.9219858156028369)
"""