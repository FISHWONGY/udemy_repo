import pandas as pd
# # Regular Expressions

'''
# Regular Expressions (sometimes called regex for short) allow a user to search for strings using almost any sort of rule they can come up with. For example, finding all capital letters in a string, or finding a phone number in a document. 
# Regular expressions are notorious for their seemingly strange syntax. This strange syntax is a byproduct of their flexibility. Regular expressions have to be able to filter out any string pattern you can imagine, which is why they have a complex string pattern format.
# Regular expressions are handled using Python's built-in **re** library. See [the docs](https://docs.python.org/3/library/re.html) for more information.
'''

# Let's begin by explaining how to search for basic patterns in a string!

# ## Searching for Basic Patterns
# Let's imagine that we have the following string:
text = "The agent's phone number is 408-555-1234. Call soon!"

# We'll start off by trying to find out if the string "phone" is inside the text string.
# Now we could quickly do this with:
'phone' in text


# But let's show the format for regular expressions, because later on we will be searching for patterns
# that won't have such a simple solution.
import re
pattern = 'phone'

re.search(pattern, text)
pattern = "NOT IN TEXT"
re.search(pattern,text)


# Now we've seen that re.search() will take the pattern, scan the text, and then returns a Match object.
# If no pattern is found, a None is returned (in Jupyter Notebook this just means that nothing is output below the cell).
# Let's take a closer look at this Match object.
pattern = 'phone'
match = re.search(pattern, text)
match

# Notice the span, there is also a start and end index information.
match.span()
match.start()
match.end()

# But what if the pattern occurs more than once?
text = "my phone is a new phone"
match = re.search("phone", text)
match.span()

# Notice it only matches the first instance. If we wanted a list of all matches, we can use .findall() method:
matches = re.findall("phone", text)
matches
len(matches)


# To get actual match objects, use the iterator:
for match in re.finditer("phone",text):
    print(match.span())


# If you wanted the actual text that matched, you can use the .group() method.
match.group()

# # Patterns
'''
# So far we've learned how to search for a basic string. What about more complex examples? Such as trying to find a telephone number in a large string of text? Or an email address?
# We could just use search method if we know the exact phone or email, but what if we don't know it? We may know the general format, and we can use that along with regular expressions to search the document for strings that match a particular pattern.
# This is where the syntax may appear strange at first, but take your time with this; often it's just a matter of looking up the pattern code.
'''
# ## Identifiers for Characters in Patterns
text = "My telephone number is 408-555-1234"
phone = re.search(r'\d\d\d-\d\d\d-\d\d\d\d', text)
phone.group()

# Now that we know the special character designations, we can use them along with quantifiers to define how many we expect.

# Let's rewrite our pattern using these quantifiers:
re.search(r'\d{3}-\d{3}-\d{4}', text)

# ## Groups
# 
# What if we wanted to do two tasks, find phone numbers, but also be able to quickly extract their area code (the first three digits). We can use groups for any general task that involves grouping together regular expressions (so that we can later break them down). 
# 
# Using the phone number example, we can separate groups of regular expressions using parentheses:
phone_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')
results = re.search(phone_pattern, text)

# The entire result
results.group()

# Can then also call by group position.
# remember groups were separated by parentheses ()
# Something to note is that group ordering starts at 1. Passing in 0 returns everything
results.group(1)
results.group(2)
results.group(3)

# We only had three groups of parentheses
results.group(4)

# ## Additional Regex Syntax
# ### Or operator |
# Use the pipe operator to have an **or** statment. For example
re.search(r"man|woman", "This man was here.")
re.search(r"man|woman", "This woman was here.")

# ### The Wildcard Character
# Use a "wildcard" as a placement that will match any character placed there. You can use a simple period **.** for this. For example:
re.findall(r".at", "The cat in the hat sat here.")
re.findall(r".at","The bat went splat")

# Notice how we only matched the first 3 letters, that is because we need a **.** for each wildcard letter. Or use the quantifiers described above to set its own rules.
re.findall(r"...at", "The bat went splat")

# However this still leads the problem to grabbing more beforehand. Really we only want words that end with "at".
# One or more non-whitespace that ends with 'at'
re.findall(r'\S+at', "The bat went splat")

# ### Starts With and Ends With
# 
# We can use the **^** to signal starts with, and the **$** to signal ends with:
# Ends with a number
re.findall(r'\d$', 'This ends with a number 2')
# Starts with a number
re.findall(r'^\d', '1 is the loneliest number.')


# Note that this is for the entire string, not individual words!
# To exclude characters, we can use the **^** symbol in conjunction with a set of brackets **[]**. Anything inside the brackets is excluded. For example:
phrase = "there are 3 numbers 34 inside 5 this sentence."
re.findall(r'[^\d]', phrase)


# To get the words back together, use a + sign 
re.findall(r'[^\d]+', phrase)


# We can use this to remove punctuation from a sentence.
test_phrase = 'This is a string! But it has punctuation. How can we remove it?'
re.findall('[^!.? ]+', test_phrase)
clean = ' '.join(re.findall('[^!.? ]+', test_phrase))
clean


# ## Brackets for Grouping
# As we showed above we can use brackets to group together options, for example if we wanted to find hyphenated words:
text = 'Only find the hypen-words in this sentence. But you do not know how long-ish they are'
re.findall(r'[\w]+-[\w]+', text)


# ## Parentheses for Multiple Options
# If we have multiple options for matching, we can use parentheses to list out these options. For Example:
# Find words that start with cat and end with one of these options: 'fish','nap', or 'claw'
text = 'Hello, would you like some catfish?'
texttwo = "Hello, would you like to take a catnap?"
textthree = "Hello, have you seen this caterpillar?"

re.search(r'cat(fish|nap|claw)', text)
re.search(r'cat(fish|nap|claw)', texttwo)

# None returned
re.search(r'cat(fish|nap|claw)',textthree)

# ### Conclusion
# Excellent work! For full information on all possible patterns, check out: https://docs.python.org/3/howto/regex.html
# ## Next up: Python Text Basics Assessment
