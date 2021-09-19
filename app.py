#!/usr/bin/env python
# coding: utf-8

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# import redis
# from flask import Flask
import streamlit as sl
import streamlit.components.v1 as components
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Flatten, TimeDistributed, Dropout, LSTMCell, RNN, Bidirectional, Concatenate, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import backend as K
from PIL import Image






sl.set_page_config(layout="wide")
img = Image.open("NomusAI.png")
sl.image(img)

sl.write('''
    # Introduction
   The law has language at its heart, so it’s not surprising that software that operates on natural language has played a role in some areas of the legal profession for a long time.NLP plays several roles in several domains of the judicial world and one of those roles is autocompletion of legal documents. And in our project we aim to implement an autocompletion system for legal documents which uses nlp works by learning human language, using context, prior queries and results to predict what attorneys need in their searches.

''')





sl.write(''' # Corpus''')



# #################



#Opening the dataset
file = open(r"holla.txt", 'r', encoding='utf-8')

corpus = [line for line in file]

#Data 
corpus[50:60]


col5,col6 = sl.columns((1,1))
col5.header('Functional Requirements')
col5.write('''
Use of cloud services: The dataset used to train the neural network is rather large, i.e. a concatenation of several pdf files, and therefore, the program cannot be executed in a local system. Use of cloud services such as Google Colab or Microsoft Azure is used to overcome the above-mentioned problem.
Input file format: The dataset provided as the input must be a .txt file
LSTM: The encoder-decoder LSTM architecture is used in this project to carry out the sequence to sequence prediction.
''')

col6.header("Non-Functional Requirements")
col6.write('''
Testability: The program should be easily testable in order to evaluate the outcome, such as the accuracy of the model.
Transparency: The working of the program should be transparent so that the user can have a basic understanding of how the model works.
Localization: Primarily meant for usage in the field of law/judiciary.
''')

sl.write('''
    # Model

''')
col12, col13 = sl.columns((1,1))
col12.write('''
    This is a sequence to sequence model which basically is a method of encoder - decoder based language processing model that maps a fixed - length inpiut with a fixed length output where the length of the input and the output may differ. 
''')
col12.text("")
col12.text("")
col12.text("")
img111 = Image.open("modelencdec.png")
col12.image(img111, use_column_width = True)
img3 = Image.open("flow.png")
col13.image(img3, use_column_width=True)


#Function to pre-process the data
def clean_special_chars(text, punct):
    for p in punct:
        text = text.replace(p, '')
    return text

#Removing the unecessary punctuations and special characters      
def preprocess(data):
    output = []
    punct = '#$%&*+-/<=>@[\\]^_`{|}~\t\n'
    for line in data:
         pline= clean_special_chars(line.lower(), punct)
         output.append(pline)
    return output  


def generate_dataset():
  
    processed_corpus = preprocess(corpus)    
    output = []
    for line in processed_corpus:
        token_list = line
        for i in range(1, len(token_list)):
            data = []
            # Adding a start and an end token to the sentence so that the model know when to start and stop predicting
            # The EOS token is important: the explicit "end" token allows the decoder to emit arbitrary-length sequences.
            # The SOS  is more important for the decoder: the decoder will progress by taking the tokens it emits as inputs 
            # (along with the embedding and hidden state), so before it has emitted anything it needs a token to start with.
            x_ngram = '<start> '+ token_list[:i+1] + ' <end>'
            y_ngram = '<start> '+ token_list[i+1:] + ' <end>'
            data.append(x_ngram)
            data.append(y_ngram) #generating pairs 
            output.append(data)
    df = pd.DataFrame(output, columns=['input','output'])
    return output, df 


class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}     #dictionary
        self.idx2word = {}     #dictionary
        self.vocab = set()     #set
        self.create_index()    #function
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))  #splitting the text where there is a space
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0  #keeping the padding tokens as 0 index
        self.idx2word[0] = "<pad>"  #keeping the 0 index to the padding tokens
        for i,word in enumerate(self.vocab):
            self.word2idx[word] = i + 1   #each word will have its own index
            self.idx2word[i+1] = word   

def max_length(t):
    return max(len(i) for i in t) #finding the maximum length for the input and output

def load_dataset():
    pairs,df = generate_dataset()

    out_lang = LanguageIndex(sp for en, sp in pairs)
    in_lang = LanguageIndex(en for en, sp in pairs)
    
    input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    max_length_in, max_length_out = max_length(input_data), max_length(output_data)
    input_data = tf.keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_length_in, padding="post")  #padding to the max_length for input_data
    output_data = tf.keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_length_out, padding="post") #padding to the max_length for output_data

    return input_data, output_data, in_lang, out_lang, max_length_in, max_length_out, df
    


# In[6]:


input_data, teacher_data, input_lang, target_lang, len_input, len_target, df = load_dataset()
# We use teacher forcing method, which works by using the actual or expected output from the training dataset at 
# the current time step y(t) as input in the next time step X(t+1), rather than the output generated by the network.

target_data = [[teacher_data[n][i+1] for i in range(len(teacher_data[n])-1)] for n in range(len(teacher_data))]



target_data = tf.keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_target, padding="post")

target_data = target_data.reshape((target_data.shape[0], target_data.shape[1], 1))

# Shuffle all of the data in unison. This training set has the longest (e.g. most complicated) data at the end,
# so a simple Keras validation split will be problematic if not shuffled.

p = np.random.permutation(len(input_data))


input_data = input_data[p]
teacher_data = teacher_data[p]
target_data = target_data[p]


# In[7]:


# Displaying the dataframe of x_ngram and y_ngram
pd.set_option('display.max_colwidth', -1)
BUFFER_SIZE = len(input_data)
BATCH_SIZE = 128
embedding_dim = 300
units = 128
vocab_in_size = len(input_lang.word2idx)
vocab_out_size = len(target_lang.word2idx)

print(vocab_out_size)
print(vocab_in_size)
df.head(10)


# In[8]:


# Create the Encoder layers first.
encoder_inputs = Input(shape=(len_input,))
encoder_emb = Embedding(input_dim=vocab_in_size, output_dim=embedding_dim)

#Using Bidirectional LSTM
encoder_lstm = Bidirectional(LSTM(units=units, return_sequences=True, return_state=True))
encoder_out, fstate_h, fstate_c, bstate_h, bstate_c = encoder_lstm(encoder_emb(encoder_inputs))
state_h = Concatenate()([fstate_h,bstate_h])
state_c = Concatenate()([bstate_h,bstate_c])
encoder_states = [state_h, state_c]


# Creating the Decoder layers.
decoder_inputs = Input(shape=(None,))
decoder_emb = Embedding(input_dim=vocab_out_size, output_dim=embedding_dim)
decoder_lstm = LSTM(units=units*2, return_sequences=True, return_state=True)
decoder_lstm_out, _, _ = decoder_lstm(decoder_emb(decoder_inputs), initial_state=encoder_states)


# Two dense layers added to this model to improve inference capabilities.
decoder_d1 = Dense(units, activation="relu")
decoder_d2 = Dense(vocab_out_size, activation="softmax")
decoder_out = decoder_d2(Dropout(rate=.2)(decoder_d1(Dropout(rate=.2)(decoder_lstm_out))))


# Creating the training model that combines the encoder and the decoder.
model = Model(inputs = [encoder_inputs, decoder_inputs], outputs= decoder_out)

# Using sparse_categorical_crossentropy so we don't have to expand decoder_out into a massive one-hot array.

model.compile(optimizer= 'adam', loss="sparse_categorical_crossentropy", metrics=['sparse_categorical_accuracy'])

col3,col4 = sl.columns(2)
col3.header('Model Summary')
img6 = Image.open("msumm.png")
col3.image(img6, use_column_width=True)



col10,col11 = sl.columns(2)
col10.header("Model Metrics - Accuracy")
img3 = Image.open("macc.png")
img4 = Image.open("mloss.png")
col10.image(img3, use_column_width=True)
col11.header("Model Metrics - Loss")
col11.image(img4, use_column_width=True)

sl.write('''
To Evaluate our model, we are using "Rouge Score" i.e. 
**"Recall Oriented Understudy Gist Evaluation".**

ROUGE is a set of metrics rather than just one. ROUGE-N measures the number of matching n-grams between our model-generated text and a 'reference'. 

Similarly, for ROUGE-1 we would be measuring the match rate of unigrams between our model output reference. 

ROUGE-2 and ROUGE-3 would use bigrams and trigrams respectively. 

ROUGE-L Measures the long common subsequence (LCS) between our model output and the 'reference. 

Each of these provides the 'Recall', 'Precision' and 'F1' score.

We are using ROUGE-1, ROUGE-2 and ROUGE-L in particular.

''')






#pip install tensorflow-estimator==2.5.*



#Using the Early stopping method

from tensorflow.keras.callbacks import EarlyStopping
earlyStop=EarlyStopping(monitor="val_loss",verbose=2,mode='min',patience=1)




# history = model.fit([input_data, teacher_data],target_data,
#                  batch_size= BATCH_SIZE,
#                  epochs=1,
#                  validation_split=0.2,
#                  steps_per_epoch = 100,
#                  callbacks = [earlyStop])



# Training Results
# plt.plot(history.history['loss'], label="Training loss")
# plt.plot(history.history['val_loss'], label="Validation loss")
# plt.show()




# Creating the encoder model from the tensors we previously declared.
encoder_model = Model(encoder_inputs, [encoder_out, state_h, state_c])

# Generate a new set of tensors for our new inference decoder.

inf_decoder_inputs = Input(shape=(None,), name="inf_decoder_inputs")
# We'll need to force feed the two state variables into the decoder each step.
state_input_h = Input(shape=(units*2,), name="state_input_h")
state_input_c = Input(shape=(units*2,), name="state_input_c")
decoder_res, decoder_h, decoder_c = decoder_lstm(
    decoder_emb(inf_decoder_inputs), 
    initial_state=[state_input_h, state_input_c])

inf_decoder_out = decoder_d2(decoder_d1(decoder_res))
inf_model = Model(inputs=[inf_decoder_inputs, state_input_h, state_input_c], 
                  outputs=[inf_decoder_out, decoder_h, decoder_c])


# Convert the given string into a vector of word IDs

def sentence_to_vector(sentence, lang):

    pre = sentence
    vec = np.zeros(len_input)
    sentence_list = [lang.word2idx[s] for s in pre.split(' ')]
    for i,w in enumerate(sentence_list):
        vec[i] = w
    return vec

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
def translate(input_sentence, infenc_model, infmodel):
    sv = sentence_to_vector(input_sentence, input_lang)
    sv = sv.reshape(1,len(sv))
    [emb_out, sh, sc] = infenc_model.predict(x=sv)
    
    i = 0
    start_vec = target_lang.word2idx["<start>"]
    stop_vec = target_lang.word2idx["<end>"]
    
    cur_vec = np.zeros((1,1))
    cur_vec[0,0] = start_vec
    cur_word = "<start>"
    output_sentence = ""

    while cur_word != "<end>" and i < (len_target-1):
        i += 1
        if cur_word != "<start>":
            output_sentence = output_sentence + " " + cur_word
        x_in = [cur_vec, sh, sc]
        [nvec, sh, sc] = infmodel.predict(x=x_in)
        cur_vec[0,0] = np.argmax(nvec[0,0])
        cur_word = target_lang.idx2word[np.argmax(nvec[0,0])]
    return output_sentence


sl.write('''
    # Try it out -
''')


test = [
    'owner may',
    'All rights',
    'will comply',
    'INCLUDING WITHOUT',
    'EACH PARTY ',
    'The term of this',
    'trustee or receiver ',
    'a '
]
  
output = []  
for t in test:  
  output.append({"Input seq":t.lower(), "Pred. Seq":translate(t.lower(), encoder_model, inf_model)})

results_df = pd.DataFrame.from_dict(output) 
results_df.head(len(test))

inp = sl.text_input("Enter something here")


output = []  

output.append({"Input seq":inp.lower(), "Pred. Seq":translate(inp.lower(), encoder_model, inf_model)})

results_df = pd.DataFrame.from_dict(output) 
results_df.head(len(test))

sl.write(results_df)


