# This unit is divided into two sections:
# * First, we'll find out what what is necessary to build an NLP system that can turn a body of text into a numerical array of *features*.
# * Next we'll show how to perform these steps using real tools.

# # Building a Natural Language Processor From Scratch
# In this section we'll use basic Python to build a rudimentary NLP system. We'll build a *corpus of documents* (two small text files), create a *vocabulary* from all the words in both documents, and then demonstrate a *Bag of Words* technique to extract features from each document.<br>

# ## Start with some documents:
# For simplicity we won't use any punctuation.

# ## Build a vocabulary
# The goal here is to build a numerical array from all the words that appear in every document. Later we'll create instances (vectors) for each individual document.
vocab = {}
i = 1

with open('1.txt') as f:
    x = f.read().lower().split()

for word in x:
    if word in vocab:
        continue
    else:
        vocab[word]=i
        i+=1

print(vocab)


with open('2.txt') as f:
    x = f.read().lower().split()

for word in x:
    if word in vocab:
        continue
    else:
        vocab[word]=i
        i+=1

print(vocab)


# Even though `2.txt` has 15 words, only 7 new words were added to the dictionary.
# ## Feature Extraction
# Now that we've encapsulated our "entire language" in a dictionary, let's perform *feature extraction* on each of our original documents:

# Create an empty vector with space for each word in the vocabulary:
one = ['1.txt']+[0]*len(vocab)
one


# map the frequencies of each word in 1.txt to our vector:
with open('1.txt') as f:
    x = f.read().lower().split()
    
for word in x:
    one[vocab[word]]+=1
    
one


# We can see that most of the words in 1.txt appear only once, although "cats" appears twice
# Do the same for the second document:
two = ['2.txt']+[0]*len(vocab)

with open('2.txt') as f:
    x = f.read().lower().split()
    
for word in x:
    two[vocab[word]]+=1


# Compare the two vectors:
print(f'{one}\n{two}')


# By comparing the vectors we see that some words are common to both, some appear only in `1.txt`, others only in `2.txt`. Extending this logic to tens of thousands of documents, we would see the vocabulary dictionary grow to hundreds of thousands of words. Vectors would contain mostly zero values, making them *sparse matrices*.

# ## Bag of Words and Tf-idf
# In the above examples, each vector can be considered a *bag of words*. By itself these may not be helpful until we consider *term frequencies*, or how often individual words appear in documents. A simple way to calculate term frequencies is to divide the number of occurrences of a word by the total number of words in the document. In this way, the number of times a word appears in large documents can be compared to that of smaller documents.
# However, it may be hard to differentiate documents based on term frequency if a word shows up in a majority of documents. To handle this we also consider *inverse document frequency*, which is the total number of documents divided by the number of documents that contain the word. In practice we convert this value to a logarithmic scale, as described [here](https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency).
# Together these terms become [**tf-idf**](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).

# ## Stop Words and Word Stems
# Some words like "the" and "and" appear so frequently, and in so many documents, that we needn't bother counting them. Also, it may make sense to only record the root of a word, say `cat` in place of both `cat` and `cats`. This will shrink our vocab array and improve performance.

# ## Tokenization and Tagging
# When we created our vectors the first thing we did was split the incoming text on whitespace with `.split()`. This was a crude form of *tokenization* - that is, dividing a document into individual words. In this simple example we didn't worry about punctuation or different parts of speech. In the real world we rely on some fairly sophisticated *morphology* to parse text appropriately.
# Once the text is divided, we can go back and *tag* our tokens with information about parts of speech, grammatical dependencies, etc. This adds more dimensions to our data and enables a deeper understanding of the context of specific documents. For this reason, vectors become ***high dimensional sparse matrices***.

# In the next section we'll use scikit-learn to perform a real-life analysis

# ___
# # Feature Extraction from Text
# In the **Scikit-learn Primer** lecture we applied a simple SVC classification model to the SMSSpamCollection dataset. We tried to predict the ham/spam label based on message length and punctuation counts. In this section we'll actually look at the text of each message and try to perform a classification based on content. We'll take advantage of some of scikit-learn's [feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) tools.
# 
# ## Load a dataset
import numpy as np
import pandas as pd

df = pd.read_csv('../TextFiles/smsspamcollection.tsv', sep='\t')
df.head()


# ## Check for missing values:
# Always a good practice.
df.isnull().sum()


# ## Take a quick look at the *ham* and *spam* `label` column:
df['label'].value_counts()
# 4825 out of 5572 messages, or 86.6%, are ham. This means that any text classification model we create has to perform **better than 86.6%** to beat random chance.</font>

# ## Split the data into train & test sets:
from sklearn.model_selection import train_test_split

X = df['message']  # this time we want to look at the text
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ## Scikit-learn's CountVectorizer
# Text preprocessing, tokenizing and the ability to filter out stopwords are all included in [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), which builds a dictionary of features and transforms documents to feature vectors.

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape


# This shows that our training set is comprised of 3733 documents, and 7082 features.

# ## Transform Counts to Frequencies with Tf-idf
# While counting words is helpful, longer documents will have higher average count values than shorter documents, even though they might talk about the same topics.
# To avoid this we can simply divide the number of occurrences of each word in a document by the total number of words in the document: these new features are called **tf** for Term Frequencies.
# Another refinement on top of **tf** is to downscale weights for words that occur in many documents in the corpus and are therefore less informative than those that occur only in a smaller portion of the corpus.
# This downscaling is called **tf–idf** for “Term Frequency times Inverse Document Frequency”.
# Both tf and tf–idf can be computed as follows using [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html):

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


# Note: the `fit_transform()` method actually performs two operations: it fits an estimator to the data and then transforms our count-matrix to a tf-idf representation.

# ## Combine Steps with TfidVectorizer
# In the future, we can combine the CountVectorizer and TfidTransformer steps into one using [TfidVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html):

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train) # remember to use the original X_train set
X_train_tfidf.shape


# ## Train a Classifier
# Here we'll introduce an SVM classifier that's similar to SVC, called [LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html). LinearSVC handles sparse input better, and scales well to large numbers of samples.

from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)


# Earlier we named our SVC classifier **svc_model**. Here we're using the more generic name **clf** (for classifier)

# ## Build a Pipeline
# Remember that only our training set has been vectorized into a full vocabulary. In order to perform an analysis on our test set we'll have to submit it to the same procedures. Fortunately scikit-learn offers a [**Pipeline**](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) class that behaves like a compound classifier.

from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),
])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  


# ## Test the classifier and display results

# Form a prediction set
predictions = text_clf.predict(X_test)


# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test,predictions))


# Print a classification report
print(metrics.classification_report(y_test,predictions))


# Print the overall accuracy
print(metrics.accuracy_score(y_test,predictions))


# Using the text of the messages, our model performed exceedingly well; it correctly predicted spam **98.97%** of the time!<br>
# Now let's apply what we've learned to a text classification project involving positive and negative movie reviews.
# 
# ## Next up: Text Classification Project
