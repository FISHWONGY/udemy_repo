# # Named Entity Recognition (NER)
# spaCy has an **'ner'** pipeline component that identifies token spans fitting a predetermined set of named entities. These are available as the `ents` property of a `Doc` object.

import spacy
nlp = spacy.load('en_core_web_sm')


# Write a function to display basic entity info:
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))
    else:
        print('No named entities found.')


doc = nlp(u'May I go to Washington, DC next May to see the Washington Monument?')

show_ents(doc)


# Here we see tokens combine to form the entities `Washington, DC`, `next May` and `the Washington Monument`

# ## Entity annotations
# `Doc.ents` are token spans with their own set of annotations.
# <table>
# <tr><td>`ent.text`</td><td>The original entity text</td></tr>
# <tr><td>`ent.label`</td><td>The entity type's hash value</td></tr>
# <tr><td>`ent.label_`</td><td>The entity type's string description</td></tr>
# <tr><td>`ent.start`</td><td>The token span's *start* index position in the Doc</td></tr>
# <tr><td>`ent.end`</td><td>The token span's *stop* index position in the Doc</td></tr>
# <tr><td>`ent.start_char`</td><td>The entity text's *start* index position in the Doc</td></tr>
# <tr><td>`ent.end_char`</td><td>The entity text's *stop* index position in the Doc</td></tr>
# </table>
# 
doc = nlp(u'Can I please borrow 500 dollars from you to buy some Microsoft stock?')

for ent in doc.ents:
    print(ent.text, ent.start, ent.end, ent.start_char, ent.end_char, ent.label_)


# ## NER Tags
# Tags are accessible through the `.label_` property of an entity.
# <table>
# <tr><th>TYPE</th><th>DESCRIPTION</th><th>EXAMPLE</th></tr>
# <tr><td>`PERSON`</td><td>People, including fictional.</td><td>*Fred Flintstone*</td></tr>
# <tr><td>`NORP`</td><td>Nationalities or religious or political groups.</td><td>*The Republican Party*</td></tr>
# <tr><td>`FAC`</td><td>Buildings, airports, highways, bridges, etc.</td><td>*Logan International Airport, The Golden Gate*</td></tr>
# <tr><td>`ORG`</td><td>Companies, agencies, institutions, etc.</td><td>*Microsoft, FBI, MIT*</td></tr>
# <tr><td>`GPE`</td><td>Countries, cities, states.</td><td>*France, UAR, Chicago, Idaho*</td></tr>
# <tr><td>`LOC`</td><td>Non-GPE locations, mountain ranges, bodies of water.</td><td>*Europe, Nile River, Midwest*</td></tr>
# <tr><td>`PRODUCT`</td><td>Objects, vehicles, foods, etc. (Not services.)</td><td>*Formula 1*</td></tr>
# <tr><td>`EVENT`</td><td>Named hurricanes, battles, wars, sports events, etc.</td><td>*Olympic Games*</td></tr>
# <tr><td>`WORK_OF_ART`</td><td>Titles of books, songs, etc.</td><td>*The Mona Lisa*</td></tr>
# <tr><td>`LAW`</td><td>Named documents made into laws.</td><td>*Roe v. Wade*</td></tr>
# <tr><td>`LANGUAGE`</td><td>Any named language.</td><td>*English*</td></tr>
# <tr><td>`DATE`</td><td>Absolute or relative dates or periods.</td><td>*20 July 1969*</td></tr>
# <tr><td>`TIME`</td><td>Times smaller than a day.</td><td>*Four hours*</td></tr>
# <tr><td>`PERCENT`</td><td>Percentage, including "%".</td><td>*Eighty percent*</td></tr>
# <tr><td>`MONEY`</td><td>Monetary values, including unit.</td><td>*Twenty Cents*</td></tr>
# <tr><td>`QUANTITY`</td><td>Measurements, as of weight or distance.</td><td>*Several kilometers, 55kg*</td></tr>
# <tr><td>`ORDINAL`</td><td>"first", "second", etc.</td><td>*9th, Ninth*</td></tr>
# <tr><td>`CARDINAL`</td><td>Numerals that do not fall under another type.</td><td>*2, Two, Fifty-two*</td></tr>
# </table>

# ___
# ## Adding a Named Entity to a Span
# Normally we would have spaCy build a library of named entities by training it on several samples of text.<br>In this case, we only want to add one value:

doc = nlp(u'Tesla to build a U.K. factory for $6 million')

show_ents(doc)


# Right now, spaCy does not recognize "Tesla" as a company.

from spacy.tokens import Span

# Get the hash value of the ORG entity label
ORG = doc.vocab.strings[u'ORG']  

# Create a Span for the new entity
new_ent = Span(doc, 0, 1, label=ORG)

# Add the entity to the existing Doc object
doc.ents = list(doc.ents) + [new_ent]


# <font color=green>In the code above, the arguments passed to `Span()` are:</font>
# -  `doc` - the name of the Doc object
# -  `0` - the *start* index position of the span
# -  `1` - the *stop* index position (exclusive)
# -  `label=ORG` - the label assigned to our entity

show_ents(doc)


# ___
# ## Adding Named Entities to All Matching Spans
# What if we want to tag *all* occurrences of "Tesla"? In this section we show how to use the PhraseMatcher to identify a series of spans in the Doc:

doc = nlp(u'Our company plans to introduce a new vacuum cleaner. '
          u'If successful, the vacuum cleaner will be our first product.')

show_ents(doc)


# Import PhraseMatcher and create a matcher object:
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)


# Create the desired phrase patterns:
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]


# Apply the patterns to our matcher object:
matcher.add('newproduct', None, *phrase_patterns)

# Apply the matcher to our Doc object:
matches = matcher(doc)

# See what matches occur:
print(matches)


# Here we create Spans from each match, and create named entities from them:
from spacy.tokens import Span

PROD = doc.vocab.strings[u'PRODUCT']

new_ents = [Span(doc, match[1], match[2], label=PROD) for match in matches]

doc.ents = list(doc.ents) + new_ents


show_ents(doc)


# ___
# ## Counting Entities
# While spaCy may not have a built-in tool for counting entities, we can pass a conditional statement into a list comprehension:
doc = nlp(u'Originally priced at $29.50, the sweater was marked down to five dollars.')

show_ents(doc)

print([ent for ent in doc.ents if ent.label_ == 'MONEY'])
print(len([ent for ent in doc.ents if ent.label_ == 'MONEY']))


# ## <font color=blue>Problem with Line Breaks</font>
# 
# <div class="alert alert-info" style="margin: 20px">There's a <a href='https://github.com/explosion/spaCy/issues/1717'>known issue</a> with <strong>spaCy v2.0.12</strong> where some linebreaks are interpreted as `GPE` entities:</div>

doc = nlp(u'Originally priced at $29.50,\nthe sweater was marked down to five dollars.')

show_ents(doc)


# ### However, there is a simple fix that can be added to the nlp pipeline

# Quick function to remove ents formed on whitespace:
# Code changed due to Spacy v3
from spacy.language import Language


@Language.component("remove_whitespace_entities")
def remove_whitespace_entities(doc):
    doc.ents = [e for e in doc.ents if not e.text.isspace()]
    return doc


# Insert this into the pipeline AFTER the ner component:
nlp.add_pipe("remove_whitespace_entities", after='ner')

# Rerun nlp on the text above, and show ents:
doc = nlp(u'Originally priced at $29.50,\nthe sweater was marked down to five dollars.')

show_ents(doc)


# For more on **Named Entity Recognition** visit https://spacy.io/usage/linguistic-features#101

# ___
# ## Noun Chunks
# `Doc.noun_chunks` are *base noun phrases*: token spans that include the noun and words describing the noun. Noun chunks cannot be nested, cannot overlap, and do not involve prepositional phrases or relative clauses.<br>
# Where `Doc.ents` rely on the **ner** pipeline component, `Doc.noun_chunks` are provided by the **parser**.

# ### `noun_chunks` components:
# <table>
# <tr><td>`.text`</td><td>The original noun chunk text.</td></tr>
# <tr><td>`.root.text`</td><td>The original text of the word connecting the noun chunk to the rest of the parse.</td></tr>
# <tr><td>`.root.dep_`</td><td>Dependency relation connecting the root to its head.</td></tr>
# <tr><td>`.root.head.text`</td><td>The text of the root token's head.</td></tr>
# </table>

doc = nlp(u"Autonomous cars shift insurance liability toward manufacturers.")

for chunk in doc.noun_chunks:
    print(chunk.text+' - '+chunk.root.text+' - '+chunk.root.dep_+' - '+chunk.root.head.text)


# ### `Doc.noun_chunks` is a  generator function
# Previously we mentioned that `Doc` objects do not retain a list of sentences, but they're available through the `Doc.sents` generator.<br>It's the same with `Doc.noun_chunks` - lists can be created if needed:

len(doc.noun_chunks)


len(list(doc.noun_chunks))


# For more on **noun_chunks** visit https://spacy.io/usage/linguistic-features#noun-chunks

# Great! Now you should be more familiar with both named entities and noun chunks. In the next section we revisit the NER visualizer.
# ## Next up: Visualizing NER
