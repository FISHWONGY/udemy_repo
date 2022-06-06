# # Visualizing Named Entities
# Besides viewing Part of Speech dependencies with `style='dep'`, **displaCy** offers a `style='ent'` visualizer:

import spacy
nlp = spacy.load('en_core_web_sm')

# Import the displaCy library
from spacy import displacy
doc = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. '
         u'By contrast, Sony sold only 7 thousand Walkman music players.')

displacy.render(doc, style='ent', jupyter=True)


# ___
# ## Viewing Sentences Line by Line
# Unlike the **displaCy** dependency parse, the NER viewer has to take in a Doc object with an `ents` attribute. For this reason, we can't just pass a list of spans to `.render()`, we have to create a new Doc from each `span.text`:
for sent in doc.sents:
    displacy.render(nlp(sent.text), style='ent', jupyter=True)


# **NOTE**: If a span does not contain any entities, displaCy will issue a harmless warning
doc2 = nlp(u'Over the last quarter Apple sold nearly 20 thousand iPods for a profit of $6 million. '
         u'By contrast, my kids sold a lot of lemonade.')


for sent in doc2.sents:
    displacy.render(nlp(sent.text), style='ent', jupyter=True)


# **WORKAROUND:** We can avert this with an additional bit of code:
for sent in doc2.sents:
    docx = nlp(sent.text)
    if docx.ents:
        displacy.render(docx, style='ent', jupyter=True)
    else:
        print(docx.text)


# ___
# ## Viewing Specific Entities
# You can pass a list of entity types to restrict the visualization:
options = {'ents': ['ORG', 'PRODUCT']}

displacy.render(doc, style='ent', jupyter=True, options=options)


# ___
# ## Customizing Colors and Effects
# You can also pass background color and gradient options:
colors = {'ORG': 'linear-gradient(90deg, #aa9cfc, #fc9ce7)', 'PRODUCT': 'radial-gradient(yellow, green)'}

options = {'ents': ['ORG', 'PRODUCT'], 'colors':colors}

displacy.render(doc, style='ent', jupyter=True, options=options)


# For more on applying CSS background colors and gradients, visit https://www.w3schools.com/css/css3_gradients.asp

# ___
# # Creating Visualizations Outside of Jupyter
# If you're using another Python IDE or writing a script, you can choose to have spaCy serve up HTML separately.
# 
# Instead of `displacy.render()`, use `displacy.serve()`:

displacy.serve(doc, style='ent', options=options)


# **After running the cell above, click the link below to view the dependency parse**
# 
# http://127.0.0.1:5000
# <br><br>
# <font color=red>**To shut down the server and return to jupyter**, interrupt the kernel either through the **Kernel** menu above, by hitting the black square on the toolbar, or by typing the keyboard shortcut `Esc`, `I`, `I`</font>

# For more on **Visualizing the entity recognizer** visit https://spacy.io/usage/visualizers#ent
# ## Next Up: Sentence Segmentation
