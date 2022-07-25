# RUN THIS CELL to perform standard imports:
import spacy
nlp = spacy.load('en_core_web_sm')

# **1. Create a Doc object from the file `owlcreek.txt`**<br>
# > HINT: Use `with open('../TextFiles/owlcreek.txt') as f:`

# Enter your code here:

with open('./udemy_repo/NLP_Natural_Language_Processing_with_Python/TextFiles/owlcreek.txt') as f:
    doc = nlp(f.read())

# Run this cell to verify it worked:

print(doc[:36])


# **2. How many tokens are contained in the file?**
print(len(doc))


# **3. How many sentences are contained in the file?**<br>HINT: You'll want to build a list first!
sents = [sent for sent in doc.sents]
print(len(sents))

# **4. Print the second sentence in the document**<br> HINT: Indexing starts at zero,
# and the title counts as the first sentence.

print(sents[1].text)


# ** 5. For each token in the sentence above, print its `text`, `POS` tag, `dep` tag and `lemma`<br>
# CHALLENGE: Have values line up in columns in the print output.**

# NORMAL SOLUTION:
for token in sents[1]:
    print(token.text, token.pos_, token.dep_, token.lemma_)


# CHALLENGE SOLUTION:
for token in sents[1]:
    print(f'{token.text:{15}} {token.pos_:{5}} {token.dep_:{10}} {token.lemma_:{15}}')


# **6. Write a matcher called 'Swimming' that finds both occurrences of the phrase
# "swimming vigorously" in the text**<br>
# HINT: You should include an `'IS_SPACE': True` pattern between the two words!

# Import the Matcher library:
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

# Create a pattern and add it to matcher:

pattern = [{'LOWER': 'swimming'}, {'IS_SPACE': True, 'OP': '*'}, {'LOWER': 'vigorously'}]

matcher.add('Swimming', [pattern])

# Create a list of matches called "found_matches" and print the list:

found_matches = matcher(doc)
print(found_matches)
# [(12881893835109366681, 1274, 1277), (12881893835109366681, 3609, 3612)]

# **7. Print the text surrounding each found match**
print(doc[1265:1290])

print(doc[3600:3615])


def surrounding(doc, start, end):
    print(doc[start-15: end+15])


surrounding(doc, 1274, 1277)
surrounding(doc, 3609, 3612)

# **EXTRA CREDIT:<br>Print the *sentence* that contains each found match**
for sent in sents:
    if found_matches[0][1] < sent.end:
        print(sent)
        break


for sent in sents:
    if found_matches[1][1] < sent.end:
        print(sent)
        break


# ### Great Job!
