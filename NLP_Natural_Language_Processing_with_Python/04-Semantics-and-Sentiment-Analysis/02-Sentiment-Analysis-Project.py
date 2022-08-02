# # Sentiment Analysis Project
# For this project, we'll perform the same type of NLTK VADER sentiment analysis, this time on our movie reviews dataset.
# 
# The 2,000 record IMDb movie review database is accessible through NLTK directly with
# However, since we already have it in a tab-delimited file we'll use that instead.

# ## Load the Data
import numpy as np
import pandas as pd

df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/moviereviews.tsv', sep='\t')
print(df.head())
# REMOVE NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  # start with an empty list

for i, lb, rv in df.itertuples():  # iterate over the DataFrame
    if type(rv) == str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

df.drop(blanks, inplace=True)

print(df['label'].value_counts())


# ## Import `SentimentIntensityAnalyzer` and create an sid object
# This assumes that the VADER lexicon has been downloaded.
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


# ## Use sid to append a `comp_score` to the dataset
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])
df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg')

print(df.head())


# ## Perform a comparison analysis between the original `label` and `comp_score`
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(df['label'], df['comp_score']))
# 0.63570691

print(classification_report(df['label'], df['comp_score']))

print(confusion_matrix(df['label'], df['comp_score']))


# So, it looks like VADER couldn't judge the movie reviews very accurately. This demonstrates one of the biggest challenges in sentiment analysis - understanding human semantics. Many of the reviews had positive things to say about a movie, reserving final judgement to the last sentence.
# ## Great Job!
