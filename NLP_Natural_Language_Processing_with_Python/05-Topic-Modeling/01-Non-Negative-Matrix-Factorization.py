# # Non-Negative Matric Factorization
# 
# Let's repeat thet opic modeling task from the previous lecture, but this time, we will use NMF instead of LDA.

# ## Data
# 
# We will be using articles scraped from NPR (National Public Radio), obtained from their website [www.npr.org](http://www.npr.org)
import pandas as pd

npr = pd.read_csv('./udemy_repo/NLP_Natural_Language_Processing_with_Python/05-Topic-Modeling/npr.csv')
print(npr.head())


# Notice how we don't have the topic of the articles! Let's use LDA to attempt to figure out clusters of the articles.

# ## Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
# **`max_df`**` : float in range [0.0, 1.0] or int, default=1.0`<br>
# When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
# 
# **`min_df`**` : float in range [0.0, 1.0] or int, default=1`<br>
# When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


dtm = tfidf.fit_transform(npr['Article'])

print(dtm)


# ## NMF
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=7, random_state=42)

# This can take awhile, we're dealing with a large amount of documents!
nmf_model.fit(dtm)


# ## Displaying Topics
print(len(tfidf.get_feature_names()))

import random

for i in range(10):
    random_word_id = random.randint(0, 54776)
    print(tfidf.get_feature_names()[random_word_id])


for i in range(10):
    random_word_id = random.randint(0, 54776)
    print(tfidf.get_feature_names()[random_word_id])


print(len(nmf_model.components_))

print(nmf_model.components_)


print(len(nmf_model.components_[0]))

single_topic = nmf_model.components_[0]

# Returns the indices that would sort this array.
single_topic.argsort()

# Word least representative of this topic
print(single_topic[18302])

# Word most representative of this topic
print(single_topic[42993])

# Top 10 words for this topic:
print(single_topic.argsort()[-10:])


top_word_indices = single_topic.argsort()[-10:]

for index in top_word_indices:
    print(tfidf.get_feature_names()[index])


# These look like business articles perhaps... Let's confirm by using .transform() on our vectorized articles to attach a label number. But first, let's view all the 10 topics found.

for index, topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# ### Attaching Discovered Topic Labels to Original Articles
print(dtm)

print(dtm.shape)

print(len(npr))

topic_results = nmf_model.transform(dtm)

print(topic_results.shape)

print(topic_results[0])

topic_results[0].round(2)

topic_results[0].argmax()


# This means that our model thinks that the first article belongs to topic #1.

# ### Combining with Original Data
print(npr.head())

topic_results.argmax(axis=1)

npr['Topic'] = topic_results.argmax(axis=1)

print(npr.head(10))


# ## Great work!
