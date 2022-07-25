# # Tokenization
# The first step in creating a `Doc` object is to break down the incoming text into component pieces or "tokens".
import spacy
nlp = spacy.load('en_core_web_sm')

# Create a string that includes opening and closing quotation marks
mystring = '"We\'re moving to L.A.!"'
print(mystring)

# Create a Doc object and explore tokens
doc = nlp(mystring)

for token in doc:
    print(token.text, end=' | ')

# ## Prefixes, Suffixes and Infixes
# spaCy will isolate punctuation that does *not* form an integral part of a word. Quotation marks, commas, and punctuation at the end of a sentence will be assigned their own token. However, punctuation that exists as part of an email address, website or numerical value will be kept as part of the token.
doc2 = nlp(u"We're here to help! Send snail-mail, email support@oursite.com or visit us at http://www.oursite.com!")

for t in doc2:
    print(t)


doc3 = nlp(u'A 5km NYC cab ride costs $10.30')

for t in doc3:
    print(t)


# ## Exceptions
# Punctuation that exists as part of a known abbreviation will be kept as part of the token.
doc4 = nlp(u"Let's visit St. Louis in the U.S. next year.")

for t in doc4:
    print(t)


# ## Counting Tokens
# `Doc` objects have a set number of tokens:
len(doc)


# ## Counting Vocab Entries
# `Vocab` objects contain a full library of items!
len(doc.vocab)

# ## Tokens can be retrieved by index position and slice
# `Doc` objects can be thought of as lists of `token` objects. As such, individual tokens can be retrieved by index position, and spans of tokens can be retrieved through slicing:
doc5 = nlp(u'It is better to give than to receive.')

# Retrieve the third token:
doc5[2]

# Retrieve three tokens from the middle:
doc5[2:5]

# Retrieve the last four tokens:
doc5[-4:]


# ## Tokens cannot be reassigned
# Although `Doc` objects can be considered lists of tokens, they do *not* support item reassignment:
doc6 = nlp(u'My dinner was horrible.')
doc7 = nlp(u'Your dinner was delicious.')

# Try to change "My dinner was horrible" to "My dinner was delicious"
doc6[3] = doc7[3]


# ___
# # Named Entities
# Going a step beyond tokens, *named entities* add another layer of context. The language model recognizes that certain words are organizational names while others are locations, and still other combinations relate to money, dates, etc. Named entities are accessible through the `ents` property of a `Doc` object.
doc8 = nlp(u'Apple to build a Hong Kong factory for $6 million')

for token in doc8:
    print(token.text, end=' | ')

print('\n----')

for ent in doc8.ents:
    print(ent)

for ent in doc8.ents:
    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))


len(doc8.ents)


# Named Entity Recognition (NER) is an important machine learning tool applied to Natural Language Processing.<br>We'll do a lot more with it in an upcoming section. For more info on **named entities** visit https://spacy.io/usage/linguistic-features#named-entities

# ---
# # Noun Chunks
# Similar to `Doc.ents`, `Doc.noun_chunks` are another object property. *Noun chunks* are "base noun phrases" – flat phrases that have a noun as their head. You can think of noun chunks as a noun plus the words describing the noun – for example, in [Sheb Wooley's 1958 song](https://en.wikipedia.org/wiki/The_Purple_People_Eater), a *"one-eyed, one-horned, flying, purple people-eater"* would be one long noun chunk.
doc9 = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")

for chunk in doc9.noun_chunks:
    print(chunk.text)


doc10 = nlp(u"Red cars do not carry higher insurance rates.")

for chunk in doc10.noun_chunks:
    print(chunk.text)


doc11 = nlp(u"He was a one-eyed, one-horned, flying, purple people-eater.")

for chunk in doc11.noun_chunks:
    print(chunk.text)


# We'll look at additional noun_chunks components besides `.text` in an upcoming section.<br>For more info on **noun_chunks** visit https://spacy.io/usage/linguistic-features#noun-chunks

# ___
# # Built-in Visualizers

# ## Visualizing the dependency parse
# Run the cell below to import displacy and display the dependency graphic
from spacy import displacy

doc = nlp(u'Apple is going to build a U.K. factory for $6 million.')
displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})


# The optional `'distance'` argument sets the distance between tokens. If the distance is made too small, text that appears beneath short arrows may become too compressed to read.

# ## Visualizing the entity recognizer
doc = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million.')
displacy.render(doc, style='ent', jupyter=True)


# ___
# ## Creating Visualizations Outside of Jupyter
# If you're using another Python IDE or writing a script, you can choose to have spaCy serve up html separately:
doc = nlp(u'This is a sentence.')
displacy.serve(doc, style='dep')


# Great! Now you should have an understanding of how tokenization divides text up into individual elements, how named entities provide context, and how certain tools help to visualize grammar rules and entity labels.
# ## Next up: Stemming
