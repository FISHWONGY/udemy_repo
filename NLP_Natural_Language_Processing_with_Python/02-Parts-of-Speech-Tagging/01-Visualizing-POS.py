import spacy
nlp = spacy.load('en_core_web_sm')

# Import the displaCy library
from spacy import displacy


# Create a simple Doc object
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")


# Render the dependency parse immediately inside Jupyter:
displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})


# The dependency parse shows the coarse POS tag for each token, as well as the **dependency tag** if given:
for token in doc:
    print(f'{token.text:{10}} {token.pos_:{7}} {token.dep_:{7}} {spacy.explain(token.dep_)}')


# ___
# # Creating Visualizations Outside of Jupyter
# If you're using another Python IDE or writing a script, you can choose to have spaCy serve up HTML separately.
# 
# Instead of `displacy.render()`, use `displacy.serve()`:

displacy.serve(doc, style='dep', options={'distance': 110})


doc2 = nlp(u"This is a sentence. This is another, possibly longer sentence.")

# Create spans from Doc.sents:
spans = list(doc2.sents)

displacy.serve(spans, style='dep', options={'distance': 110})


# ## Customizing the Appearance
# Besides setting the distance between tokens, you can pass other arguments to the `options` parameter:
# For a full list of options visit https://spacy.io/api/top-level#displacy_options

options = {'distance': 110, 'compact': 'True', 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'}

displacy.serve(doc, style='dep', options=options)


# Great! Now you should be familiar with visualizing spaCy's dependency parse. For more info on **displaCy** visit https://spacy.io/usage/visualizers
# <br>In the next section we'll look at Named Entity Recognition, followed by displaCy's NER visualizer.
# 
# ### Next Up: Named Entity Recognition
