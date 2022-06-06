# spaCy Basics
# Import spaCy and load the language library
import spacy
nlp = spacy.load('en_core_web_sm')

# Create a Doc object
doc = nlp(u'Tesla is looking at buying U.S. startup for $6 million')

# Print each token separately
for token in doc:
    print(token.text, token.pos_, token.dep_)


# This doesn't look very user-friendly, but right away we see some interesting things happen:
# 1. Tesla is recognized to be a Proper Noun, not just a word at the start of a sentence
# 2. U.S. is kept together as one entity (we call this a 'token')

# spaCy Objects
# After importing the spacy module in the cell above we loaded a **model** and named it `nlp`.<br>Next we created a **Doc** object by applying the model to our text, and named it `doc`.<br>spaCy also builds a companion **Vocab** object that we'll cover in later sections.<br>The **Doc** object that holds the processed text is our focus here.

# ___
# # Pipeline
nlp.pipeline

nlp.pipe_names


# ___
# ## Tokenization
# The first step in processing text is to split up all the component parts (words & punctuation) into "tokens". These tokens are annotated inside the Doc object to contain descriptive information. We'll go into much more detail on tokenization in an upcoming lecture. For now, let's look at another example:
doc2 = nlp(u"Tesla isn't   looking into startups anymore.")

for token in doc2:
    print(token.text, token.pos_, token.dep_)


# Notice how `isn't` has been split into two tokens. spaCy recognizes both the root verb `is` and the negation attached to it. Notice also that both the extended whitespace and the period at the end of the sentence are assigned their own tokens.
# It's important to note that even though `doc2` contains processed information about each token, it also retains the original text:
doc2

doc2[0]
type(doc2)

# ## Part-of-Speech Tagging (POS)
# The next step after splitting the text up into tokens is to assign parts of speech. In the above example, `Tesla` was recognized to be a ***proper noun***. Here some statistical modeling is required. For example, words that follow "the" are typically nouns.
# For a full list of POS Tags visit https://spacy.io/api/annotation#pos-tagging
doc2[0].pos_

# ___
# ## Dependencies
# We also looked at the syntactic dependencies assigned to each token. `Tesla` is identified as an `nsubj` or the ***nominal subject*** of the sentence.
# For a full list of Syntactic Dependencies visit https://spacy.io/api/annotation#dependency-parsing
doc2[0].dep_

# To see the full name of a tag use `spacy.explain(tag)`
spacy.explain('PROPN')
spacy.explain('nsubj')


# ___
# ## Additional Token Attributes
# We'll see these again in upcoming lectures. For now we just want to illustrate some of the other information that spaCy assigns to tokens:
# Lemmas (the base form of the word):
print(doc2[4].text)
print(doc2[4].lemma_)

# Simple Parts-of-Speech & Detailed Tags:
print(doc2[4].pos_)
print(doc2[4].tag_ + ' / ' + spacy.explain(doc2[4].tag_))

# Word Shapes:
print(doc2[0].text+': '+doc2[0].shape_)
print(doc[5].text+' : '+doc[5].shape_)

# Boolean Values:
print(doc2[0].is_alpha)
print(doc2[0].is_stop)

# ___
# ## Spans
# Large Doc objects can be hard to work with at times. A **span** is a slice of Doc object in the form `Doc[start:stop]`.
doc3 = nlp(u'Although commmonly attributed to John Lennon from his song "Beautiful Boy", \
the phrase "Life is what happens to us while we are making other plans" was written by \
cartoonist Allen Saunders and published in Reader\'s Digest in 1957, when Lennon was 17.')

life_quote = doc3[16:30]
print(life_quote)
type(life_quote)


# In upcoming lectures we'll see how to create Span objects using `Span()`. This will allow us to assign additional information to the Span.

# ___
# ## Sentences
# Certain tokens inside a Doc object may also receive a "start of sentence" tag. While this doesn't immediately build a list of sentences, these tags enable the generation of sentence segments through `Doc.sents`. Later we'll write our own segmentation rules.
doc4 = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')

for sent in doc4.sents:
    print(sent)

doc4[6].is_sent_start


# ## Next up: Tokenization
