import numpy as np
from collections import defaultdict
import pandas as pd
from keras.utils import np_utils
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

####
import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding

from chessRL import make_matrix
import copy

path = '../data/games.csv'
df = pd.read_csv(path)
df = df.dropna(subset=['moves'])

class DeepModel:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.999    # discount rate #0.95
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9 #0.99
        self.learning_rate = 1e-5 #0.001
        self.model = self._build_model()
        #self.target_model = self._build_model()
        #self.update_target_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(5, input_dim=self.state_size, activation='relu')) #24
        model.add(Dense(5, activation='relu')) #24
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy'])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        ##With probability e select a random action at
        if np.random.rand() <= self.epsilon:
            #return 3 #lerning with stockfish
            return random.randrange(self.action_size)
        ##otherwise select at = maxa Q∗(φ(st), a; θ)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def predict(self, state):
        act_values = self.model.predict(state)
        max_val = np.argmax(act_values[0])
        return max_val,act_values

    def train(self,X,y, epochs=10): # train 
        #history = self.model.fit(state, target, epochs=5, verbose=0)
        history = self.model.fit(np.array(X),y, epochs=epochs, verbose=0)
        return history

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  #plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  #plt.ylim([0,20])
  plt.legend()
  plt.show()
        
state_size = 66
action_size = 7
# Pawn, Knight, Bishop, Rook, Queen, King
pieces_dict = {
    'p': 0,     #Pawn
    'N': 1,     #Knight
    'B': 2,     #Bishop
    'R': 3,     #Rook
    'Q': 4,     #Queen
    'K': 5,     #King
    'O': 6,     #King - Castle
    }

model = DeepModel(state_size, action_size)
with open("../models/opening_encoder", "rb") as f:
    opp_enc = pickle.load(f)

X = np.load("../data/chess_deep_next_piece_X_train.npy")
y = np.load("../data/chess_deep_next_piece_y_train.npy")
#X = np.array(X)
#encoder = LabelEncoder()
#encoder.fit(y)
#encoded_Y = encoder.transform(y)
#dummy_y = np_utils.to_categorical(encoded_Y)
#np.save("../data/chess_deep_next_piece_y_train", dummy_y)

history = model.train(X,y,30)
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#plot_history(history)

model.save("../models/chess_deep_next_piece.h5")

print(len(X))
print(len(y))
