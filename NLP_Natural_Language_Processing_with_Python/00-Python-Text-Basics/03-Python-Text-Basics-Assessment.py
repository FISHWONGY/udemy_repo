from IPython import get_ipython
# ## f-Strings
# #### 1. Print an f-string that displays `NLP stands for Natural Language Processing` using the variables provided.abbr = 'NLP'
full_text = 'Natural Language Processing'

# Enter your code here:


# ## Files
# #### 2. Create a file in the current working directory called `contacts.txt` by running the cell below:
get_ipython().run_cell_magic('writefile', 'contacts.txt', 'First_Name Last_Name, Title, Extension, Email\n')


# #### 3. Open the file and use .read() to save the contents of the file to a string called `fields`.  Make sure the file is closed at the end.
# Write your code here:
with open('contacts.txt') as c:
    fields = c.read()


    
# Run fields to see the contents of contacts.txt:
fields


# ## Working with PDF Files
# #### 4. Use PyPDF2 to open the file `Business_Proposal.pdf`. Extract the text of page 2.
# Perform import
import PyPDF2

# Open the file as a binary object
f = open('Business_Proposal.pdf','rb')

# Use PyPDF2 to read the text of the file
pdf_reader = PyPDF2.PdfFileReader(f)


# Get the text from page 2 (CHALLENGE: Do this in one step!)
page_two_text = pdf_reader.getPage(1).extractText()



# Close the file
f.close()

# Print the contents of page_two_text
print(page_two_text)


# #### 5. Open the file `contacts.txt` in append mode. Add the text of page 2 from above to `contacts.txt`.
# #### CHALLENGE: See if you can remove the word "AUTHORS:"
#
# Simple Solution:
with open('contacts.txt','a+') as c:
    c.write(page_two_text)
    c.seek(0)
    print(c.read())



# CHALLENGE Solution (re-run the %%writefile cell above to obtain an unmodified contacts.txt file):
with open('contacts.txt','a+') as c:
    c.write(page_two_text[8:])
    c.seek(0)
    print(c.read())




# ## Regular Expressions
# #### 6. Using the `page_two_text` variable created above, extract any email addresses that were contained in the file `Business_Proposal.pdf`.
import re
# Enter your regex pattern here. This may take several tries!
pattern = r'\w+@\w+.\w{3}'

re.findall(pattern, page_two_text)


# ### Great job!
