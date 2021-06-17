import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
import random

import MCTS
from min_max_tree import calculate_min_max_tree, calculate_min_max_tree2

######

from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
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
        

def create_model(vocabulary_size, seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(10,return_sequences=True))
    model.add(LSTM(10))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))
    # compiling the network
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

#load LSTM model
t_len = 3
model_name = f"../models/lstm_model/chessML_len{t_len}.h5"
token_name = f"../models/lstm_model/tokenizer_len{t_len}.pkl"

with open(token_name, 'rb') as handle:
        tokenizer = pickle.load(handle)

stats = pd.read_csv('../logs/train_data_log.csv')
vocabulary_size = int(stats['vocabulary_size'][1])
seq_len = int(stats['seq_len'][1])

lstm_model = create_model(vocabulary_size, seq_len)    
lstm_model.load_weights(model_name)

def lstm_predict(seq_len,model, tokenizer, board, debug = False):
    next_moves = []
    input_text = board
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')
    #print(encoded_text, pad_encoded)

    for i in (model.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
      pred_word = tokenizer.index_word[i]
      next_moves.append(pred_word)
      if debug:
          print("Next word suggestion:",pred_word)

    return next_moves

pieces_dict = {
    '0': 'p',     #Pawn
    '1': 'N',     #Knight
    '2': 'B',     #Bishop
    '3': 'R',     #Rook
    '4': 'Q',     #Queen
    '5': 'K',     #King
    '6': 'O',     #King - Castle
    }


env = gym.make('Chess-v0')
state = env.reset()

cnt = 0
string = "e4 c5 Nc3 Nc6 Qf3 e5"
string = "e4 e5 Nf3 Qf6 c3 d5 d3 Bd7 exd5 Qd6 c4 Qb4+ Qd2 Qxd2+ Nfxd2 Bc8 Nf3 Bb4+ Bd2 Bd6"
string = "e4"
game = string.split(' ')
for move in game:
    #move = state.san(action)
    action = state.push_san(move)
                
    #take action
    state,reward,done,_ = env.step(action)
    cnt +=1

print(env.render())
print(state.status())


with open("../models/opening_encoder", "rb") as f:
    opp_enc = pickle.load(f)
    
model = DeepModel(66,7)
model.load("../models/chess_deep_next_piece.h5")

model_result = DeepModel(66,3)
model_result.load("../models/chess_deep_result.h5")

opp = opp_enc.transform(['B50'])[0]

history_move = []
for i in string.split(' '):
    history_move.append(i)

engine = chess.engine.SimpleEngine.popen_uci("../stockfish")
df1 = pd.read_pickle("../data/pgn/chess_deep_memory.pkl")

while not state.is_game_over():
        print(env.render())
        board_txt = state.fen().split()[0]
        board_encoded = ''.join(str(ord(c)) for c in board_txt)
        obs = make_matrix(state)
        
        if cnt % 2 == 0:
            print("White") #board, player, oppening
            player = 'white'
            #board_state = [np.reshape(obs, [1, 64]), 1, 0]
            board_state = list(np.reshape(obs, [1, 64])[0])
            board_state.append(1)
            board_state.append(opp)
            board_state = np.array(board_state)
            #print(env.render())
            #print(obs)
            #print(board_state)
        else:
            print("Black")
            player = 'black'
            #board_state = [np.reshape(obs, [1, 64]), 0, 0]
            board_state = list(np.reshape(obs, [1, 64])[0])
            board_state.append(0)
            board_state.append(opp)
            board_state = np.array(board_state)

        legal_moves = [state.san(x) for x in env.legal_moves]

        #Next Piece predict
        X = board_state
        max_val,act_values_pieces = model.predict(np.reshape(X, [1, 66]))
        print("Next Piece: ",max_val,act_values_pieces )

        #Result predict
        X = board_state
        max_val,act_values = model_result.predict(np.reshape(X, [1, 66]))
        print("Result: ",max_val,act_values)
        
        act_values = list(act_values[0])
        pieces_ordered = []
        for i in act_values:
            max_val = max(act_values)
            pieces_ordered.append(act_values.index(max_val))
            act_values[act_values.index(max_val)] = -1
            
        pieces_ordered = [pieces_dict[str(x)] for x in pieces_ordered]
        pieces_ordered = pieces_ordered[:4]

        #LSTM ML predict
        string_moves = ''
        for i,v in enumerate(history_move):
            if i == 0:
                string_moves = v
            else:
                string_moves += ' ' + v
        
        next_moves = lstm_predict(seq_len, lstm_model, tokenizer, string_moves)
        next_moves = [x.capitalize() if len(x) > 2 else x for x in next_moves]
        print("LSTM: ",next_moves)

        #MinMaxTree
        #best_move = calculate_min_max_tree(state, env, player, depth=2, mode='stockfish', engine=engine)
        #if cnt % 2 == 0:
        #    best_move = calculate_min_max_tree2(state, env, player, depth=2, mode='stockfish', engine=engine,list_moves=next_moves, next_pieces=list(act_values_pieces[0]))
        #    print("MinMax Tree: ", best_move)

        #Player
        if cnt % 2 == 0:
            player = 'white'
            df2 = df1[df1['board'] == str(obs)]
            next_moves = []
            if len(df1) >=0:
                next_moves = list(df2['next_move'])
                print('Next moves Player: ', next_moves)
            
            best_move = calculate_min_max_tree2(state, env, player, depth=0, mode='stockfish', engine=engine,list_moves=next_moves, next_pieces=list(act_values_pieces[0]))
            print("Player: ", best_move)
            
        
        #MCTS
        #root,selected_node,weights,material_adv = MCTS.calculate_MCTS(state = state, env = env,filter_prop='next_piece' ,filter_moves=pieces_ordered, par='material_adv')
        #print("MCTS: ",selected_node.parent_action)
        #print(selected_node.steps)
        #print(selected_node.material_adv)
        #print(material_adv)

        if cnt % 2 == 0:
            move = best_move
        else:
            #legal_moves = [state.san(x) for x in env.legal_moves]
            #move = legal_moves[random.randrange(len(legal_moves))]
            move = input("Insira um valor: ")
        history_move.append(move)
        action = state.push_san(move)
        cnt +=1
        #take action
        try:
            state,reward,done,_ = env.step(action)
        except:
            break
