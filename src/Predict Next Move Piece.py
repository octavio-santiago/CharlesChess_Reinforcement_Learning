import numpy as np
from collections import defaultdict
import pandas as pd
from keras.utils import np_utils

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

opp_enc = LabelEncoder()
opp_enc.fit(df['opening_eco'])
#opp_list = opp_enc.transform(df['opening_eco'])

#opp = df['opening_eco'][0]
X = []
y = []

env = gym.make('Chess-v0')

for idx,game in enumerate(df['moves']):
    opp = opp_enc.transform([df['opening_eco'][idx]])[0]
    game = game.split(' ')
    state = env.reset()
    #print(env.render())

    for cnt, move in enumerate(game):
        if cnt == len(game) -1:
            break
        
        board_txt = state.fen().split()[0]
        board_encoded = ''.join(str(ord(c)) for c in board_txt)
        obs = make_matrix(state)
        
        
        if cnt % 2 == 0:
            #print("White") #board, player, oppening
            #board_state = [np.reshape(obs, [1, 64]), 1, 0]
            board_state = list(np.reshape(obs, [1, 64])[0])
            board_state.append(1)
            board_state.append(opp)
            board_state = np.array(board_state)
            #print(env.render())
            #print(obs)
            #print(board_state)
        else:
            #print("Black")
            #board_state = [np.reshape(obs, [1, 64]), 0, 0]
            board_state = list(np.reshape(obs, [1, 64])[0])
            board_state.append(0)
            board_state.append(opp)
            board_state = np.array(board_state)

        X.append(board_state)
        
        next_p = game[cnt+1]
        if not next_p[0].isupper():
            ans = pieces_dict['p']
        else:
            ans = pieces_dict[next_p[0]]
           
        y.append(ans)
        action = state.push_san(move)     
        #take action
        try:
            state,reward,done,_ = env.step(action)
        except:
            break

X = np.array(X)
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)
np.save("../data/chess_deep_next_piece_X_train", X)
np.save("../data/chess_deep_next_piece_y_train", y)

with open("../models/opening_encoder", "wb") as f:
    pickle.dump(opp_enc, f)
    
print(len(X))
print(len(y))
