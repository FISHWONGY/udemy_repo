from IPython import get_ipython
# jupyter nbconvert notebook.ipynb --to python

name = 'Fred'

# Using the old .format() method:
print('His name is {var}.'.format(var=name))

# Using f-strings:
print(f'His name is {name}.')

print(f'His name is {name!r}')

d = {'a': 123, 'b': 456}

print(f'Address: {d["a"]} Main Street')

d = {'a':123,'b':456}

print(f"Address: {d['a']} Main Street")

library = [('Author', 'Topic', 'Pages'), ('Twain', 'Rafting', 601), ('Feynman', 'Physics', 95), ('Hamilton', 'Mythology', 144)]

for book in library:
    print(f'{book[0]:{10}} {book[1]:{8}} {book[2]:{7}}')

for book in library:
    print(f'{book[0]:{10}} {book[1]:{10}} {book[2]:.>{7}}')  # here .> was added


# Date Formatting
from datetime import datetime

today = datetime(year=2018, month=1, day=27)

print(f'{today:%B %d, %Y}')

# Open the text.txt file we created earlier
my_file = open('./test.txt')

my_file

# We can now read the file
my_file.read()

# But what happens if we try to read it again?
my_file.read()

# Seek to the start of file (index 0)
my_file.seek(0)

# Now read again
my_file.read()

'''.readlines()
You can read a file line by line using the readlines method. Use caution with large files, 
since everything will be held in memory. 
We will learn how to iterate over large files later in the course.'''

# Readlines returns a list of the lines in the file
my_file.seek(0)
my_file.readlines()

# Writing to a File
# By default, the open() function will only allow us to read the file.
# We need to pass the argument 'w' to write over the file. For example:


# Add a second argument to the function, 'w' which stands for write.
# Passing 'w+' lets us read and write to the file

my_file = open('test.txt', 'w+')

# Write to the file
my_file.write('This is a new first line')

# Read the file
my_file.seek(0)
my_file.read()

my_file.close()  # always do this when you're done with a file

'''
Appending to a File
Passing the argument 'a' opens the file and puts the pointer at the end, so anything written is appended. 
Like 'w+', 'a+' lets us read and write to a file. If the file does not exist, one will be created.
'''

my_file = open('test.txt', 'a+')
my_file.write('\nThis line is being appended to test.txt')
my_file.write('\nAnd another line here.')

my_file.seek(0)
print(my_file.read())

my_file.close()

'''
Aliases and Context Managers
You can assign temporary variable names as aliases, 
and manage the opening and closing of files automatically using a context manager:
'''
with open('test.txt', 'r') as txt:
    first_line = txt.readlines()[0]

print(first_line)

# Iterating through a File
with open('test.txt', 'r') as txt:
    for line in txt:
        print(line, end='')  # the end='' argument removes extra linebreaks


