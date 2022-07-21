import pandas as pd

# In[52]:


quora = pd.read_csv('quora_questions.csv')


# In[53]:


quora.head()


# # Preprocessing
# 
# #### Task: Use TF-IDF Vectorization to create a vectorized document term matrix. You may want to explore the max_df and min_df parameters.

# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[41]:


tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


# In[42]:


dtm = tfidf.fit_transform(quora['Question'])


# In[43]:


dtm


# # Non-negative Matrix Factorization
# 
# #### TASK: Using Scikit-Learn create an instance of NMF with 20 expected components. (Use random_state=42)..

# In[44]:


from sklearn.decomposition import NMF


# In[48]:


nmf_model = NMF(n_components=20,random_state=42)


# In[49]:


nmf_model.fit(dtm)


# #### TASK: Print our the top 15 most common words for each of the 20 topics.

# In[50]:


for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# #### TASK: Add a new column to the original quora dataframe that labels each question into one of the 20 topic categories.

# In[54]:


quora.head()


# In[55]:


topic_results = nmf_model.transform(dtm)


# In[56]:


topic_results.argmax(axis=1)

quora['Topic'] = topic_results.argmax(axis=1)

quora.head(10)


# # Great job!
