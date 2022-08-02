# # Sentiment Analysis
# Now that we've seen word vectors we can start to investigate sentiment analysis. The goal is to find commonalities between documents, with the understanding that similarly *combined* vectors should correspond to similar sentiments.
# 
# While the scope of sentiment analysis is very broad, we will focus our work in two ways.
# 
# ### 1. Polarity classification
# We won't try to determine if a sentence is objective or subjective, fact or opinion. Rather, we care only if the text expresses a *positive*, *negative* or *neutral* opinion.
# ### 2. Document level scope
# We'll also try to aggregate all of the sentences in a document or paragraph, to arrive at an overall opinion.
# ### 3. Coarse analysis
# We won't try to perform a fine-grained analysis that would determine the degree of positivity/negativity. That is, we're not trying to guess how many stars a reviewer awarded, just whether the review was positive or negative.

# ## Broad Steps:
# * First, consider the text being analyzed. A model trained on paragraph-long movie reviews might not be effective on tweets. Make sure to use an appropriate model for the task at hand.
# * Next, decide the type of analysis to perform. In the previous section on text classification we used a bag-of-words technique that considered only single tokens, or *unigrams*. Some rudimentary sentiment analysis models go one step further, and consider two-word combinations, or *bigrams*. In this section, we'd like to work with complete sentences, and for this we're going to import a trained NLTK lexicon called *VADER*.


import nltk
# nltk.download('vader_lexicon')
# This is due to be fixed in an upcoming NLTK release. For now, if you want to avoid it you can (optionally) install the NLTK twitter library with<br>
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


# VADER's `SentimentIntensityAnalyzer()` takes in a string and returns a dictionary of scores in each of four categories:
# * negative
# * neutral
# * positive
# * compound *(computed by normalizing the scores above)*

a = 'This was a good movie.'
sid.polarity_scores(a)


a = 'This was the best, most awesome movie EVER MADE!!!'
sid.polarity_scores(a)


a = 'This was the worst film to ever disgrace the screen.'
sid.polarity_scores(a)


# ## Use VADER to analyze Amazon Reviews
# For this exercise we're going to apply `SentimentIntensityAnalyzer` to a dataset of 10,000 Amazon reviews. Like our movie reviews datasets, these are labeled as either "pos" or "neg". At the end we'll determine the accuracy of our sentiment analysis with VADER.
import pandas as pd

df = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/amazonreviews.tsv', sep='\t')
df.head()
df['label'].value_counts()


# ### Clean the data (optional):
# Recall that our moviereviews.tsv file contained empty records. Let's check to see if any exist in amazonreviews.tsv.
# REMOVE NaN VALUES AND EMPTY STRINGS:
df.dropna(inplace=True)

blanks = []  # start with an empty list

for i, lb, rv in df.itertuples():  # iterate over the DataFrame
    if type(rv) == str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list

df.drop(blanks, inplace=True)


print(df['label'].value_counts())


# In this case there were no empty records. Good!

# ## Let's run the first review through VADER
sid.polarity_scores(df.loc[0]['review'])

print(df.loc[0]['label'])


# Great! Our first review was labeled "positive", and earned a positive compound score.

# ## Adding Scores and Labels to the DataFrame
# In this next section we'll add columns to the original DataFrame to store polarity_score dictionaries, extracted compound scores, and new "pos/neg" labels derived from the compound score. We'll use this last column to perform an accuracy test.
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))

print(df.head())

df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])

print(df.head())

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

print(df.head())


# ## Report on Accuracy
# Finally, we'll use scikit-learn to determine how close VADER came to our original 10,000 labels.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy_score(df['label'], df['comp_score'])
print(classification_report(df['label'], df['comp_score']))
print(confusion_matrix(df['label'], df['comp_score']))


# This tells us that VADER correctly identified an Amazon review as "positive" or "negative" roughly 71% of the time.
# ## Up Next: Sentiment Analysis Project
