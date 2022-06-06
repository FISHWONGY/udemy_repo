# # Text Generation with Neural Networks

# ## Functions for Processing Text
def read_file(filepath):
    
    with open(filepath) as f:
        str_text = f.read()
    
    return str_text

read_file('moby_dick_four_chapters.txt')


# ### Tokenize and Clean Text
import spacy
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

nlp.max_length = 1198623


def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']


d = read_file('melville-moby_dick.txt')
tokens = separate_punc(d)

tokens

len(tokens)

4431/25


# ## Create Sequences of Tokens
# organize into sequences of tokens
train_len = 25 + 1 # 50 training words , then one target word

# Empty list of sequences
text_sequences = []

for i in range(train_len, len(tokens)):
    
    # Grab train_len# amount of characters
    seq = tokens[i-train_len:i]
    
    # Add to list of sequences
    text_sequences.append(seq)


' '.join(text_sequences[0])


# In[ ]:


' '.join(text_sequences[1])


' '.join(text_sequences[2])

len(text_sequences)


# # Keras

# ### Keras Tokenization
from keras.preprocessing.text import Tokenizer

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_sequences)
sequences = tokenizer.texts_to_sequences(text_sequences)

sequences[0]


tokenizer.index_word


for i in sequences[0]:
    print(f'{i} : {tokenizer.index_word[i]}')

tokenizer.word_counts


vocabulary_size = len(tokenizer.word_counts)


# ### Convert to Numpy Matrix
import numpy as np

sequences = np.array(sequences)

sequences


# # Creating an LSTM based model

import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding


def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, 25, input_length=seq_len))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(150))
    model.add(Dense(150, activation='relu'))

    model.add(Dense(vocabulary_size, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   
    model.summary()
    
    return model


# ### Train / Test Split

from keras.utils import to_categorical

sequences


# First 49 words
sequences[:,:-1]


# last Word
sequences[:,-1]

X = sequences[:, :-1]

y = sequences[:, -1]


y = to_categorical(y, num_classes=vocabulary_size+1)


seq_len = X.shape[1]

seq_len


# ### Training the Model

# define model
model = create_model(vocabulary_size+1, seq_len)



from pickle import dump, load

# fit model
model.fit(X, y, batch_size=128, epochs=300, verbose=1)


# save the model to file
model.save('epochBIG.h5')
# save the tokenizer
dump(tokenizer, open('epochBIG', 'wb'))


# # Generating New Text

from random import randint
from pickle import load
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences


def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):
    '''
    INPUTS:
    model : model that was trained on text data
    tokenizer : tokenizer that was fit on text data
    seq_len : length of training sequence
    seed_text : raw string text to serve as the seed
    num_gen_words : number of words to be generated by model
    '''
    
    # Final Output
    output_text = []
    
    # Intial Seed Sequence
    input_text = seed_text
    
    # Create num_gen_words
    for i in range(num_gen_words):
        
        # Take the input text string and encode it to a sequence
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        
        # Pad sequences to our trained rate (50 words in the video)
        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
        
        # Predict Class Probabilities for each word
        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]
        
        # Grab word
        pred_word = tokenizer.index_word[pred_word_ind] 
        
        # Update the sequence of input text (shifting one over with the new word)
        input_text += ' ' + pred_word
        
        output_text.append(pred_word)
        
    # Make it look like a sentence.
    return ' '.join(output_text)


# ### Grab a random seed sequence
text_sequences[0]

import random
random.seed(101)
random_pick = random.randint(0,len(text_sequences))


random_seed_text = text_sequences[random_pick]

random_seed_text


seed_text = ' '.join(random_seed_text)

seed_text

generate_text(model, tokenizer, seq_len, seed_text=seed_text, num_gen_words=50)


# ### Exploring Generated Sequence

full_text = read_file('moby_dick_four_chapters.txt')

for i, word in enumerate(full_text.split()):
    if word == 'inkling':
        print(' '.join(full_text.split()[i-20:i+20]))
        print('\n')


# # Great Job!
