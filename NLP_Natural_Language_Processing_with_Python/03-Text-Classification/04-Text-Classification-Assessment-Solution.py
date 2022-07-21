import numpy as np
import pandas as pd

df = pd.read_csv('../TextFiles/moviereviews2.tsv', sep='\t')
df.head()

# ### Task #2: Check for missing values:

# In[3]:


# Check for NaN values:
df.isnull().sum()


# In[2]:


# Check for whitespace strings (it's OK if there aren't any!):
blanks = []  # start with an empty list

for i, lb, rv in df.itertuples():  # iterate over the DataFrame
    if type(rv) == str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
        
len(blanks)


# ### Task #3:  Remove NaN values:

# In[4]:


df.dropna(inplace=True)


# ### Task #4: Take a quick look at the `label` column:

# In[5]:


df['label'].value_counts()


# ### Task #5: Split the data into train & test sets:
# You may use whatever settings you like. To compare your results to the solution notebook, use `test_size=0.33, random_state=42`

# In[6]:


from sklearn.model_selection import train_test_split

X = df['review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# ### Task #6: Build a pipeline to vectorize the date, then train and fit a model
# You may use whatever model you like. To compare your results to the solution notebook, use `LinearSVC`.

# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()), ])

# Feed the training data through the pipeline
text_clf.fit(X_train, y_train)  


# ### Task #7: Run predictions and analyze the results

# In[8]:


# Form a prediction set
predictions = text_clf.predict(X_test)


# In[9]:


# Report the confusion matrix
from sklearn import metrics
print(metrics.confusion_matrix(y_test, predictions))


# In[10]:


# Print a classification report
print(metrics.classification_report(y_test, predictions))


# In[11]:


# Print the overall accuracy
print(metrics.accuracy_score(y_test, predictions))


# ## Great job!
