# # Semantics and Word Vectors
# Sometimes called "opinion mining", [Wikipedia](https://en.wikipedia.org/wiki/Sentiment_analysis) defines ***sentiment analysis*** as
# <div class="alert alert-info" style="margin: 20px">"the use of natural language processing ... to systematically identify, extract, quantify, and study affective states and subjective information.<br>
# Generally speaking, sentiment analysis aims to determine the attitude of a speaker, writer, or other subject with respect to some topic or the overall contextual polarity or emotional reaction to a document, interaction, or event."</div>
# 
# Up to now we've used the occurrence of specific words and word patterns to perform test classifications. In this section we'll take machine learning even further, and try to extract intended meanings from complex phrases. Some simple examples include:
# * Python is relatively easy to learn.
# * That was the worst movie I've ever seen.
# 
# However, things get harder with phrases like:
# * I do not dislike green eggs and ham. (requires negation handling)
# 
# The way this is done is through complex machine learning algorithms like [word2vec](https://en.wikipedia.org/wiki/Word2vec). The idea is to create numerical arrays, or *word embeddings* for every word in a large corpus. Each word is assigned its own vector in such a way that words that frequently appear together in the same context are given vectors that are close together. The result is a model that may not know that a "lion" is an animal, but does know that "lion" is closer in context to "cat" than "dandelion".
# 
# It is important to note that *building* useful models takes a long time - hours or days to train a large corpus - and that for our purposes it is best to import an existing model rather than take the time to train our own.
# 

# # Word Vectors
# Word vectors - also called *word embeddings* - are mathematical descriptions of individual words such that words that appear frequently together in the language will have similar values. In this way we can mathematically derive *context*. As mentioned above, the word vector for "lion" will be closer in value to "cat" than to "dandelion".

# ## Vector values
# So what does a word vector look like? Since spaCy employs 300 dimensions, word vectors are stored as 300-item arrays.
# 
# Note that we would see the same set of values with **en_core_web_md** and **en_core_web_lg**, as both were trained using the [word2vec](https://en.wikipedia.org/wiki/Word2vec) family of algorithms.
import spacy
nlp = spacy.load('en_core_web_md')  # make sure to use a larger model!

nlp(u'lion').vector


doc = nlp(u'The quick brown fox jumped over the lazy dogs.')

doc.vector


# ## Identifying similar vectors
# Create a three-token Doc object:
tokens = nlp(u'lion cat pet')

# Iterate through token combinations:
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# Note that order doesn't matter. `token1.similarity(token2)` has the same value as `token2.similarity(token1)
# #### To view this as a table:

# For brevity, assign each token a name
a, b, c = tokens

# Display as a Markdown table (this only works in Jupyter!)
from IPython.display import Markdown, display
display(Markdown(f'<table><tr><th></th><th>{a.text}</th><th>{b.text}</th><th>{c.text}</th></tr>\
<tr><td>**{a.text}**</td><td>{a.similarity(a):{.4}}</td><td>{b.similarity(a):{.4}}</td><td>{c.similarity(a):{.4}}</td></tr>\
<tr><td>**{b.text}**</td><td>{a.similarity(b):{.4}}</td><td>{b.similarity(b):{.4}}</td><td>{c.similarity(b):{.4}}</td></tr>\
<tr><td>**{c.text}**</td><td>{a.similarity(c):{.4}}</td><td>{b.similarity(c):{.4}}</td><td>{c.similarity(c):{.4}}</td></tr>'))


# As expected, we see the strongest similarity between "cat" and "pet", the weakest between "lion" and "pet", and some similarity between "lion" and "cat". A word will have a perfect (1.0) similarity with itself.
# 
# If you're curious, the similarity between "lion" and "dandelion" is very small:

nlp(u'lion').similarity(nlp(u'dandelion'))


# ### Opposites are not necessarily different
# Words that have opposite meaning, but that often appear in the same *context* may have similar vectors.

# Create a three-token Doc object:
tokens = nlp(u'like love hate')

# Iterate through token combinations:
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity(token2))


# ## Vector norms
# It's sometimes helpful to aggregate 300 dimensions into a [Euclidian (L2) norm](https://en.wikipedia.org/wiki/Norm_%28mathematics%29#Euclidean_norm), computed as the square root of the sum-of-squared-vectors. This is accessible as the `.vector_norm` token attribute. Other helpful attributes include `.has_vector` and `.is_oov` or *out of vocabulary*.
# 
# For example, our 685k vector library may not have the word "[nargle](https://en.wikibooks.org/wiki/Muggles%27_Guide_to_Harry_Potter/Magic/Nargle)". To test this:

tokens = nlp(u'dog cat nargle')

for token in tokens:
    print(token.text, token.has_vector, token.vector_norm, token.is_oov)


# Indeed we see that "nargle" does not have a vector, so the vector_norm value is zero, and it identifies as *out of vocabulary*.

# ## Vector arithmetic
# Believe it or not, we can actually calculate new vectors by adding & subtracting related vectors. A famous example suggests
# <pre>"king" - "man" + "woman" = "queen"</pre>
# Let's try it out!

from scipy import spatial

cosine_similarity = lambda x, y: 1 - spatial.distance.cosine(x, y)

king = nlp.vocab['king'].vector
man = nlp.vocab['man'].vector
woman = nlp.vocab['woman'].vector

# Now we find the closest vector in the vocabulary to the result of "man" - "woman" + "queen"
new_vector = king - man + woman
computed_similarities = []

for word in nlp.vocab:
    # Ignore words without vectors and mixed-case words:
    if word.has_vector:
        if word.is_lower:
            if word.is_alpha:
                similarity = cosine_similarity(new_vector, word.vector)
                computed_similarities.append((word, similarity))

computed_similarities = sorted(computed_similarities, key=lambda item: -item[1])

print([w[0].text for w in computed_similarities[:10]])


# So in this case, "king" was still closer than "queen" to our calculated vector, although "queen" did show up!

# ## Next up: Sentiment Analysis
