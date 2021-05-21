from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from keras.utils import to_categorical
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences


path = r'C:\Users\Octavio\Desktop\Projetos Python\Chess-RL\games.csv'
df = pd.read_csv(path)
df = df.dropna(subset=['moves'])
#df = df[(df['winner'] == 'white') & (df['victory_status'] == 'mate')].reset_index()
'''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['moves'])

sequence_data = tokenizer.texts_to_sequences(df['moves'])[0]

vocab_size = len(tokenizer.word_index) + 1

sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
    
sequences = np.array(sequences)

X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])
    
X = np.array(X)
y = np.array(y)

y = to_categorical(y, num_classes=vocab_size)

train_inputs = X
train_targets = y
seq_len = train_inputs.shape[1]'''

def preproc(df, t_len,token_name):
    text_sequences = []
    for j in range(0,len(df['moves'])):
        cleaned = df['moves'][j]
        tokens = word_tokenize(cleaned)
        #print(tokens)
        train_len = t_len


        for i in range(train_len,len(tokens)):
          seq = tokens[i-train_len:i]
          text_sequences.append(seq)
      
        sequences = {}
        count = 1

        for i in range(len(tokens)):
          if tokens[i] not in sequences:
            sequences[tokens[i]] = count
            count += 1
        
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    with open(token_name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sequences = tokenizer.texts_to_sequences(text_sequences)

    print(sequences)

    #vocabulary size increased by 1 for the cause of padding
    vocabulary_size = len(tokenizer.word_counts)+1
    n_sequences = np.empty([len(sequences),train_len], dtype='int32')

    for i in range(len(sequences)):
      n_sequences[i] = sequences[i]
      
    train_inputs = n_sequences[:,:-1]
    train_targets = n_sequences[:,-1]
    train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
    seq_len = train_inputs.shape[1]

    #print(train_inputs)
    stats = pd.DataFrame({'vocabulary_size':[vocabulary_size], 'seq_len':[seq_len] })
    stats.to_csv('train_data_log.csv')
    return vocabulary_size, seq_len, train_inputs, train_targets

#Train

def create_model(vocabulary_size, seq_len, train_inputs, train_targets):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(10,return_sequences=True)) #50
    model.add(LSTM(10))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    # compiling the network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def train(vocabulary_size, seq_len, train_inputs, train_targets,name):
    model = create_model(vocabulary_size, seq_len, train_inputs, train_targets)
    history = model.fit(train_inputs,train_targets,epochs=5,verbose=1)
    model.save_weights(name)
    #save vocabulary_size, seq_len
    return history

def predict(seq_len,model, tokenizer):
    input_text = input("Insert the board moves: ").strip()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
    print(encoded_text, pad_encoded)

    for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
      pred_word = tokenizer.index_word[i]
      print("Next word suggestion:",pred_word)

      
#MAIN
t_len = 3
model_name = f"chessML_len{t_len}.h5"
token_name = f"tokenizer_len{t_len}.pkl"
vocabulary_size, seq_len, train_inputs, train_targets = preproc(df, t_len,token_name)

mode = 'train'
#load
if mode == 'predict':
    with open(token_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    model = create_model(vocabulary_size, seq_len, train_inputs, train_targets)    
    model.load_weights(model_name)
    predict(seq_len,model, tokenizer)
else:
    history = train(vocabulary_size, seq_len, train_inputs, train_targets,model_name)
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print(hist.tail())
      







