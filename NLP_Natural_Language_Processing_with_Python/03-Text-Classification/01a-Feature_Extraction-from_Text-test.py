import pandas as pd
df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/smsspamcollection.tsv', sep='\t')
label_dict = {"ham": 0, "spam": 1}
df = df.replace({"label": label_dict})
from sklearn.model_selection import train_test_split

X = df['message']  # this time we want to look at the text
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set
print(X_train_tfidf.shape)

from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

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

