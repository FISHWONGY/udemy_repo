# # Sentence Segmentation
# In **spaCy Basics** we saw briefly how Doc objects are divided into sentences. In this section we'll learn how sentence segmentation works, and how to set our own segmentation rules.

import spacy
nlp = spacy.load('en_core_web_sm')

# From Spacy Basics:
doc = nlp(u'This is the first sentence. This is another sentence. This is the last sentence.')

for sent in doc.sents:
    print(sent)


# ### `Doc.sents` is a generator
# It is important to note that `doc.sents` is a *generator*. That is, a Doc is not segmented until `doc.sents` is called. This means that, where you could print the second Doc token with `print(doc[1])`, you can't call the "second Doc sentence" with `print(doc.sents[1])`:

print(doc[1])

print(doc.sents[1])


# However, you *can* build a sentence collection by running `doc.sents` and saving the result to a list:
doc_sents = [sent for sent in doc.sents]
print(doc_sents)


# **NOTE**: `list(doc.sents)` also works. We show a list comprehension as it allows you to pass in conditionals.

# Now you can access individual sentences:
print(doc_sents[1])


# ### `sents` are Spans
# At first glance it looks like each `sent` contains text from the original Doc object. In fact they're just Spans with start and end token pointers.
type(doc_sents[1])

print(doc_sents[1].start, doc_sents[1].end)


# ## Adding Rules
# spaCy's built-in `sentencizer` relies on the dependency parse and end-of-sentence punctuation to determine segmentation rules. We can add rules of our own, but they have to be added *before* the creation of the Doc object, as that is where the parsing of segment start tokens happens:

# Parsing the segmentation start tokens happens during the nlp pipeline
doc2 = nlp(u'This is a sentence. This is a sentence. This is a sentence.')

for token in doc2:
    print(token.is_sent_start, ' '+token.text)


# >Notice we haven't run `doc2.sents`, and yet `token.is_sent_start` was set to True on two tokens in the Doc.

# Let's add a semicolon to our existing segmentation rules. That is, whenever the sentencizer encounters a semicolon, the next token should start a new segment.

# SPACY'S DEFAULT BEHAVIOR
doc3 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc3.sents:
    print(sent)


# ADD A NEW RULE TO THE PIPELINE
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc


nlp.add_pipe(set_custom_boundaries, before='parser')

print(nlp.pipe_names)


# The new rule has to run before the document is parsed. Here we can either pass the argument `before='parser'` or `first=True`.

# Re-run the Doc object creation:
doc4 = nlp(u'"Management is doing things right; leadership is doing the right things." -Peter Drucker')

for sent in doc4.sents:
    print(sent)


# And yet the new rule doesn't apply to the older Doc object:
for sent in doc3.sents:
    print(sent)


# ### Why not change the token directly?
# Why not simply set the `.is_sent_start` value to True on existing tokens?
# Find the token we want to change:
doc3[7]


# Try to change the .is_sent_start attribute:
doc3[7].is_sent_start = True


# spaCy refuses to change the tag after the document is parsed to prevent inconsistencies in the data

# ## Changing the Rules
# In some cases we want to *replace* spaCy's default sentencizer with our own set of rules. In this section we'll see how the default sentencizer breaks on periods. We'll then replace this behavior with a sentencizer that breaks on linebreaks.
nlp = spacy.load('en_core_web_sm')  # reset to the original

mystring = u"This is a sentence. This is another.\n\nThis is a \nthird sentence."

# SPACY DEFAULT BEHAVIOR:
doc = nlp(mystring)

for sent in doc.sents:
    print([token.text for token in sent])


# CHANGING THE RULES
from spacy.pipeline import SentenceSegmenter

def split_on_newlines(doc):
    start = 0
    seen_newline = False
    for word in doc:
        if seen_newline:
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text.startswith('\n'): # handles multiple occurrences
            seen_newline = True
    yield doc[start:]      # handles the last group of tokens


sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newlines)
nlp.add_pipe(sbd)


# While the function `split_on_newlines` can be named anything we want, it's important to use the name `sbd` for the SentenceSegmenter.</font>

doc = nlp(mystring)
for sent in doc.sents:
    print([token.text for token in sent])


# <font color=green>Here we see that periods no longer affect segmentation, only linebreaks do. This would be appropriate when working with a long list of tweets, for instance.</font>
# ## Next Up: POS Assessment
