# # Stop Words
# Words like "a" and "the" appear so frequently that they don't require tagging as thoroughly as nouns, verbs and modifiers. We call these *stop words*, and they can be filtered from the text to be processed. spaCy holds a built-in list of some 305 English stop words.
import spacy
nlp = spacy.load('en_core_web_sm')


# Print the set of spaCy's default stop words (remember that sets are unordered):
print(nlp.Defaults.stop_words)


len(nlp.Defaults.stop_words)


# ## To see if a word is a stop word
nlp.vocab['myself'].is_stop


nlp.vocab['mystery'].is_stop


# ## To add a stop word
# There may be times when you wish to add a stop word to the default set. Perhaps you decide that `'btw'` (common shorthand for "by the way") should be considered a stop word.
# Add the word to the set of stop words. Use lowercase!
nlp.Defaults.stop_words.add('btw')

# Set the stop_word tag on the lexeme
nlp.vocab['btw'].is_stop = True


len(nlp.Defaults.stop_words)
nlp.vocab['btw'].is_stop


# <font color=green>When adding stop words, always use lowercase. Lexemes are converted to lowercase before being added to **vocab**.</font>

# ## To remove a stop word
# Alternatively, you may decide that `'beyond'` should not be considered a stop word.

# Remove the word from the set of stop words
nlp.Defaults.stop_words.remove('beyond')

# Remove the stop_word tag from the lexeme
nlp.vocab['beyond'].is_stop = False


len(nlp.Defaults.stop_words)
nlp.vocab['beyond'].is_stop


# Great! Now you should be able to access spaCy's default set of stop words, and add or remove stop words as needed.
# ## Next up: Vocabulary and Matching
