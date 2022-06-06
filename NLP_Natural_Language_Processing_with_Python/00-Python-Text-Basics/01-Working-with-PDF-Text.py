
# Keep in mind that not every PDF file can be read with this library. PDFs that are too blurry,have a special encoding, encrypted, or maybe just created with a particular program that doesn't work well with PyPDF2 won't be able to be read. If you find yourself in this situation, try using the libraries linked above, but keep in mind, these may also not work. The reason for this is because of the many different parameters for a PDF and how non-standard the settings can be, text could be shown as an image instead of a utf-8 encoding. There are many parameters to consider in this aspect.
# As far as PyPDF2 is concerned, it can only read the text from a PDF document, it won't be able to grab images or other media files from a PDF.

# ## Working with PyPDF2
import PyPDF2

# First we open a pdf, then create a reader object for it. Notice how we use the binary method of reading , 'rb', instead of just 'r'.

# Notice we read it as a binary with 'rb'
f = open('/Users/yuawong/Documents/GitHub/udemy_repo/NLP_Natural_Language_Processing_with_Python/00-Python-Text-Basics/'
         'US_Declaration.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(f)
pdf_reader.numPages
page_one = pdf_reader.getPage(0)

# We can then extract the text:
page_one_text = page_one.extractText()
page_one_text
f.close()


# ## Adding to PDFs
# We can not write to PDFs using Python because of the differences between the single string type of Python, and the variety of fonts, placements, and other parameters that a PDF could have.
# What we *can* do is copy pages and append pages to the end.
f = open('/Users/yuawong/Documents/GitHub/udemy_repo/NLP_Natural_Language_Processing_with_Python/00-Python-Text-Basics/'
         'US_Declaration.pdf', 'rb')
pdf_reader = PyPDF2.PdfFileReader(f)
first_page = pdf_reader.getPage(0)
pdf_writer = PyPDF2.PdfFileWriter()
pdf_writer.addPage(first_page)
pdf_output = open("/Users/yuawong/Documents/GitHub/udemy_repo/NLP_Natural_Language_Processing_with_Python/"
                  "00-Python-Text-Basics/Some_New_Doc.pdf", "wb")
pdf_writer.write(pdf_output)
pdf_output.close()
f.close()


# Now we have copied a page and added it to another new document!
# ## Simple Example
# Let's try to grab all the text from this PDF file:
f = open('US_Declaration.pdf', 'rb')

# List of every page's text.
# The index will correspond to the page number.
pdf_text = [0]  # zero is a placehoder to make page 1 = index 1

pdf_reader = PyPDF2.PdfFileReader(f)

for p in range(pdf_reader.numPages):
    
    page = pdf_reader.getPage(p)
    
    pdf_text.append(page.extractText())

f.close()


pdf_text
print(pdf_text[2])


# ### Excellent work! 
# That is all for PyPDF2 for now, remember that this won't work with every PDF file and is limited in its scope to only the text of PDFs.
# ## Next up: Regular Expressions
