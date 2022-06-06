# # Latent Dirichlet Allocation

# ## Data
# 
# We will be using articles from NPR (National Public Radio), obtained from their website [www.npr.org](http://www.npr.org)
import pandas as pd
npr = pd.read_csv('npr.csv')
npr.head()


# Notice how we don't have the topic of the articles! Let's use LDA to attempt to figure out clusters of the articles.

# ## Preprocessing
from sklearn.feature_extraction.text import CountVectorizer


# When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
# When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')

dtm = cv.fit_transform(npr['Article'])

dtm


# ## LDA
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7,random_state=42)

# This can take awhile, we're dealing with a large amount of documents!
LDA.fit(dtm)

# ## Showing Stored Words
len(cv.get_feature_names())

import random


for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])


for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])


# ### Showing Top Words Per Topic
len(LDA.components_)

LDA.components_

len(LDA.components_[0])


single_topic = LDA.components_[0]

# Returns the indices that would sort this array.
single_topic.argsort()

# Word least representative of this topic
single_topic[18302]

# Word most representative of this topic
single_topic[42993]

# Top 10 words for this topic:
single_topic.argsort()[-10:]


top_word_indices = single_topic.argsort()[-10:]


for index in top_word_indices:
    print(cv.get_feature_names()[index])


# These look like business articles perhaps... Let's confirm by using .transform() on our vectorized articles to attach a label number. But first, let's view all the 10 topics found.

for index, topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# ### Attaching Discovered Topic Labels to Original Articles
dtm


dtm.shape

len(npr)


topic_results = LDA.transform(dtm)

topic_results.shape


topic_results[0]


topic_results[0].round(2)


topic_results[0].argmax()


# This means that our model thinks that the first article belongs to topic #1.

# ### Combining with Original Data
npr.head()

topic_results.argmax(axis=1)

npr['Topic'] = topic_results.argmax(axis=1)

npr.head(10)


# ## Great work!
