# # Part of Speech Basics
# The challenge of correctly identifying parts of speech is summed up nicely in the [spaCy docs](https://spacy.io/usage/linguistic-features):
# In this section we'll take a closer look at coarse POS tags (noun, verb, adjective) and fine-grained tags (plural noun, past-tense verb, superlative adjective).

import spacy
nlp = spacy.load('en_core_web_sm')


# Create a simple Doc object
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")


# ## View token tags
# Recall that you can obtain a particular token by its index position.
# * To view the coarse POS tag use `token.pos_`
# * To view the fine-grained tag use `token.tag_`
# * To view the description of either type of tag use `spacy.explain(tag)`

# Print the full text:
print(doc.text)


# Print the fifth word and associated tags:
print(doc[4].text, doc[4].pos_, doc[4].tag_, spacy.explain(doc[4].tag_))


# We can apply this technique to the entire Doc object:
for token in doc:
    print(f'{token.text:{10}} {token.pos_:{8}} {token.tag_:{6}} {spacy.explain(token.tag_)}')


# ## Coarse-grained Part-of-speech Tags
# Every token is assigned a POS Tag from the following list:
# Tokens are subsequently given a fine-grained tag as determined by morphology:
# For a current list of tags for all languages visit https://spacy.io/api/annotation#pos-tagging

# ## Working with POS Tags
# In the English language, the same string of characters can have different meanings, even within the same sentence. For this reason, morphology is important. **spaCy** uses machine learning algorithms to best predict the use of a token in a sentence. Is *"I read books on NLP"* present or past tense? Is *wind* a verb or a noun?

doc = nlp(u'I read books on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')


doc = nlp(u'I read a book on NLP.')
r = doc[1]

print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')


# In the first example, with no other cues to work from, spaCy assumed that ***read*** was present tense.<br>In the second example the present tense form would be ***I am reading a book***, so spaCy assigned the past tense.

# ## Counting POS Tags
# The `Doc.count_by()` method accepts a specific token attribute as its argument, and returns a frequency count of the given attribute as a dictionary object. Keys in the dictionary are the integer values of the given attribute ID, and values are the frequency. Counts of zero are not included.

doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

# Count the frequencies of different coarse-grained POS tags:
POS_counts = doc.count_by(spacy.attrs.POS)
POS_counts


# This isn't very helpful until you decode the attribute ID:
doc.vocab[83].text


# ### Create a frequency list of POS tags from the entire document
# Since `POS_counts` returns a dictionary, we can obtain a list of keys with `POS_counts.items()`.<br>By sorting the list we have access to the tag and its count, in order.
for k, v in sorted(POS_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')


# Count the different fine-grained tags:
TAG_counts = doc.count_by(spacy.attrs.TAG)

for k, v in sorted(TAG_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{4}}: {v}')


# Count the different dependencies:
DEP_counts = doc.count_by(spacy.attrs.DEP)

for k, v in sorted(DEP_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{4}}: {v}')


# Here we've shown `spacy.attrs.POS`, `spacy.attrs.TAG` and `spacy.attrs.DEP`.<br>Refer back to the **Vocabulary and Matching** lecture from the previous section for a table of **Other token attributes**.

# ___
# ## Fine-grained POS Tag Examples
# These are some grammatical examples (shown in **bold**) of specific fine-grained tags. We've removed punctuation and rarely used tags:

# ### Up Next: Visualizing POS
