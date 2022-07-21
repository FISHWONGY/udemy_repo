import spacy
nlp = spacy.load('en_core_web_md')

# In[2]:


# Choose the words you wish to compare, and obtain their vectors
word1 = nlp.vocab['wolf'].vector
word2 = nlp.vocab['dog'].vector
word3 = nlp.vocab['cat'].vector


# In[3]:


# Import spatial and define a cosine_similarity function
from scipy import spatial

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)


# In[4]:


# Write an expression for vector arithmetic
# For example: new_vector = word1 - word2 + word3
new_vector = word1 - word2 + word3


# In[5]:


# List the top ten closest vectors in the vocabulary to the result of the expression above
computed_similarities = []

for word in nlp.vocab:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])


# #### CHALLENGE: Write a function that takes in 3 strings, performs a-b+c arithmetic, and returns a top-ten result

# In[6]:


def vector_math(a,b,c):
    new_vector = nlp.vocab[a].vector - nlp.vocab[b].vector + nlp.vocab[c].vector
    computed_similarities = []

    for word in nlp.vocab:
        if word.has_vector:
            if word.is_lower:
                if word.is_alpha:
                    similarity = cosine_similarity(new_vector, word.vector)
                    computed_similarities.append((word, similarity))

    computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

    return [w[0].text for w in computed_similarities[:10]]


# In[7]:


# Test the function on known words:
vector_math('king','man','woman')


# ## Task #2: Perform VADER Sentiment Analysis on your own review
# Write code that returns a set of SentimentIntensityAnalyzer polarity scores based on your own written review.

# In[8]:


# Import SentimentIntensityAnalyzer and create an sid object
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()


# In[9]:


# Write a review as one continuous string (multiple sentences are ok)
review = 'This movie portrayed real people, and was based on actual events.'


# In[10]:


# Obtain the sid scores for your review
sid.polarity_scores(review)


# ### CHALLENGE: Write a function that takes in a review and returns a score of "Positive", "Negative" or "Neutral"

# In[11]:


def review_rating(string):
    scores = sid.polarity_scores(string)
    if scores['compound'] == 0:
        return 'Neutral'
    elif scores['compound'] > 0:
        return 'Positive'
    else:
        return 'Negative'


# In[12]:


# Test the function on your review above:
review_rating(review)


# ## Great job!
