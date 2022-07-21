# RUN THIS CELL to perform standard imports:
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy

# **1. Create a Doc object from the file `peterrabbit.txt`**<br>
# > HINT: Use `with open('../TextFiles/peterrabbit.txt') as f:`

# In[2]:


with open('../TextFiles/peterrabbit.txt') as f:
    doc = nlp(f.read())


# **2. For every token in the third sentence, print the token text, the POS tag, the fine-grained TAG tag, and the description of the fine-grained tag.**

# In[3]:


# Enter your code here:

for token in list(doc.sents)[2]:
    print(f'{token.text:{12}} {token.pos_:{6}} {token.tag_:{6}} {spacy.explain(token.tag_)}')


# **3. Provide a frequency list of POS tags from the entire document**

# In[4]:


POS_counts = doc.count_by(spacy.attrs.POS)

for k,v in sorted(POS_counts.items()):
    print(f'{k}. {doc.vocab[k].text:{5}}: {v}')


# **4. CHALLENGE: What percentage of tokens are nouns?**<br>
# HINT: the attribute ID for 'NOUN' is 91

# In[5]:


percent = 100*POS_counts[91]/len(doc)

print(f'{POS_counts[91]}/{len(doc)} = {percent:{.4}}%')


# **5. Display the Dependency Parse for the third sentence**

# In[6]:


displacy.render(list(doc.sents)[2], style='dep', jupyter=True, options={'distance': 110})


# **6. Show the first two named entities from Beatrix Potter's *The Tale of Peter Rabbit* **

# In[7]:


for ent in doc.ents[:2]:
    print(ent.text+' - '+ent.label_+' - '+str(spacy.explain(ent.label_)))


# **7. How many sentences are contained in *The Tale of Peter Rabbit*?**

# In[8]:


len([sent for sent in doc.sents])


# **8. CHALLENGE: How many sentences contain named entities?**

# In[9]:


list_of_sents = [nlp(sent.text) for sent in doc.sents]
list_of_ners = [doc for doc in list_of_sents if doc.ents]
len(list_of_ners)


# **9. CHALLENGE: Display the named entity visualization for `list_of_sents[0]` from the previous problem**

# In[10]:


displacy.render(list_of_sents[0], style='ent', jupyter=True)


# ### Great Job!
