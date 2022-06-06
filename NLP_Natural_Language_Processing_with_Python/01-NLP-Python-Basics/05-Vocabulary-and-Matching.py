# # Vocabulary and Matching
# So far we've seen how a body of text is divided into tokens, and how individual tokens are parsed and tagged with parts of speech, dependencies and lemmas.
# 
# In this section we will identify and label specific phrases that match patterns we can define ourselves. 

# ## Rule-based Matching
# spaCy offers a rule-matching tool called `Matcher` that allows you to build a library of token patterns, then match those patterns against a Doc object to return a list of found matches. You can match on any part of the token including text and annotations, and you can add multiple patterns to the same matcher.

import spacy
nlp = spacy.load('en_core_web_sm')


# Import the Matcher library
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)


# ### Creating patterns
# In literature, the phrase 'solar power' might appear as one word or two, with or without a hyphen. In this section we'll develop a matcher named 'SolarPower' that finds all three:
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]

matcher.add('SolarPower', None, pattern1, pattern2, pattern3)


# Let's break this down:
# * `pattern1` looks for a single token whose lowercase text reads 'solarpower'
# * `pattern2` looks for two adjacent tokens that read 'solar' and 'power' in that order
# * `pattern3` looks for three adjacent tokens, with a middle token that can be any punctuation.<font color=green>*</font>
# 
# <font color=green>\* Remember that single spaces are not tokenized, so they don't count as punctuation.</font>
# <br>Once we define our patterns, we pass them into `matcher` with the name 'SolarPower', and set *callbacks* to `None` (more on callbacks later).

# ### Applying the matcher to a Doc object

doc = nlp(u'The Solar Power industry continues to grow as demand \
for solarpower increases. Solar-power cars are gaining popularity.')


found_matches = matcher(doc)
print(found_matches)


# `matcher` returns a list of tuples. Each tuple contains an ID for the match, with start & end tokens that map to the span `doc[start:end]`


for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  # get string representation
    span = doc[start:end]                    # get the matched span
    print(match_id, string_id, start, end, span.text)


# The `match_id` is simply the hash value of the `string_ID` 'SolarPower'

# ### Setting pattern options and quantifiers
# You can make token rules optional by passing an `'OP':'*'` argument. This lets us streamline our patterns list:

# Redefine the patterns:
pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2)


found_matches = matcher(doc)
print(found_matches)


# This found both two-word patterns, with and without the hyphen!
# 
# The following quantifiers can be passed to the `'OP'` key:


# ### Be careful with lemmas!
# If we wanted to match on both 'solar power' and 'solar powered', it might be tempting to look for the *lemma* of 'powered' and expect it to be 'power'. This is not always the case! The lemma of the *adjective* 'powered' is still 'powered':

pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LEMMA': 'power'}] # CHANGE THIS PATTERN

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2)


doc2 = nlp(u'Solar-powered energy runs solar-powered cars.')


found_matches = matcher(doc2)
print(found_matches)


# The matcher found the first occurrence because the lemmatizer treated 'Solar-powered' as a verb, but not the second as it considered it an adjective.<br>For this case it may be better to set explicit token patterns.</font>

pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solarpowered'}]
pattern4 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP':'*'}, {'LOWER': 'powered'}]

# Remove the old patterns to avoid duplication:
matcher.remove('SolarPower')

# Add the new set of patterns to the 'SolarPower' matcher:
matcher.add('SolarPower', None, pattern1, pattern2, pattern3, pattern4)


found_matches = matcher(doc2)
print(found_matches)


# ## Other token attributes
# Besides lemmas, there are a variety of token attributes we can use to determine matching rules:
# ### Token wildcard
# You can pass an empty dictionary `{}` as a wildcard to represent **any token**. For example, you might want to retrieve hashtags without knowing what might follow the `#` character:
# >`[{'ORTH': '#'}, {}]`

# ___
# ## PhraseMatcher
# In the above section we used token patterns to perform rule-based matching. An alternative - and often more efficient - method is to match on terminology lists. In this case we use PhraseMatcher to create a Doc object from a list of phrases, and pass that into `matcher` instead.
import spacy
nlp = spacy.load('en_core_web_sm')


# Import the PhraseMatcher library
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)


# For this exercise we're going to import a Wikipedia article on *Reaganomics*<br>
# Source: https://en.wikipedia.org/wiki/Reaganomics

with open('../TextFiles/reaganomics.txt', encoding='utf8') as f:
    doc3 = nlp(f.read())


# First, create a list of match phrases:
phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']

# Next, convert each phrase to a Doc object:
phrase_patterns = [nlp(text) for text in phrase_list]

# Pass each Doc object into matcher (note the use of the asterisk!):
matcher.add('VoodooEconomics', None, *phrase_patterns)

# Build a list of matches:
matches = matcher(doc3)


# (match_id, start, end)
matches


# <font color=green>The first four matches are where these terms are used in the definition of Reaganomics:</font>
doc3[:70]


# ## Viewing Matches
# There are a few ways to fetch the text surrounding a match. The simplest is to grab a slice of tokens from the doc that is wider than the match:
doc3[665:685]  # Note that the fifth match starts at doc3[673]


doc3[2975:2995]  # The sixth match starts at doc3[2985]


# Another way is to first apply the `sentencizer` to the Doc, then iterate through the sentences to the match point:

# Build a list of sentences
sents = [sent for sent in doc3.sents]

# In the next section we'll see that sentences contain start and end token values:
print(sents[0].start, sents[0].end)


# Iterate over the sentence list until the sentence end value exceeds a match start value:
for sent in sents:
    if matches[4][1] < sent.end:  # this is the fifth match, that starts at doc3[673]
        print(sent)
        break


# For additional information visit https://spacy.io/usage/linguistic-features#section-rule-based-matching
# ## Next Up: NLP Basics Assessment
