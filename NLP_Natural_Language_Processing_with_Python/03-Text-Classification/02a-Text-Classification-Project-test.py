import pandas as pd
df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/moviereviews.tsv', sep='\t')
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
    text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words=stopwords)),
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

