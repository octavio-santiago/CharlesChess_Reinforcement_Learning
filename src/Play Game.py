import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt

import MCTS
from min_max_tree import calculate_min_max_tree, calculate_min_max_tree2

######

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
from chessRL import get_moves_by_similarity
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
string = "e4 e5 Nf3 Qf6 c3 d5 d3 Bd7 exd5 Qd6 c4 Qb4+ Qd2 Qxd2+ Nfxd2 Bc8 Nf3 Bb4+ Bd2 Bd6"
string = "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 Re1 Nd6 Nxe5 Be7"
#string = "e4 c5 Nc3 Nc6 Qf3 e5"
string = "e4 c5 Nf3 e6"
#string = "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 Re1 Nd6 Nxe5 Be7 Bf1 Nxe5 Rxe5 O-O d4 Bf6 Re1 Re8 c3 Rxe1 Qxe1 Ne8 Bf4 d5 Bd3 g6 Nd2 Ng7 Nf3 Bf5 Bxf5 Nxf5 Qe2 Qd7 Ne5 Qe7 h3 Re8 Re1 c6 Qd1 Qd8 Qd2 h5 Re2 Ng7 Nd3 Rxe2 Qxe2 Qe7 Be5 Nf5 a3 Bg7 Kf1 Bxe5 Nxe5 Qf6 Nd7 Ng3+ Ke1 Qd8 fxg3 Qxd7 Qe5 a5 Kf2 Qd8 h4 a4 Qe2 Kf8 Kg1 Qe7 Qf2 g5 Kf1 Kg7 Kg1 Qf6 Qe1 Kg6 Kh2 gxh4 gxh4 Qf4+ Kh3 Qg4+ Kh2 f6 Qe7 Qxh4+ Kg1 b6 Qe8+ Kg7 Qd8 Qe4 Qc7+ Kg6 Qxc6 h4 Qc8 Kg5 Qb7 Qb1+ Kh2 Qf5 Qa8 Qf4+ Kh1 Qe4 Qb7 Kf4 Qxb6 Qf5 Qd6+ Ke3 Kg1 Qb1+ Kh2 Qf5 Kg1 Qf2+ Kh2 Qf5 c4 Kxd4 c5 Qe5+ Kh1 Qe1+ Kh2 Qg3+ Kg1 Qe1+ Kh2 Qe5+ Kh1 f5 Qd7 f4 c6 h3 Qxh3 Kc5 Qc8 Qh5+ Kg1 Qd1+ Kf2 Qc2+ Kg1 Qe4 b4+ axb3 Qf8+ Kxc6 Qc8+ Kd6 Qd8+ Kc5 Qf8+ Kb5 Qb8+ Kc4 Qb4+ Kd3 Qxb3+ Kd2 Qb2+ Kd1 Qa1+ Kc2 Qa2+ Kc3 a4 Qe3+ Kh2 Qg3+ Kh1 Kb4 a5 Qe1+ Kh2 Qg3+ Kg1 Qe3+ Kh2 Qg3+ Kg1 Qe3+ Kh2 d4 a6 d3 Qd5 Qg3+ Kh1 Qe3 Qb7+ Kc4 Qc7+ Kb3 Qb8+ Kc2 Qc7+ Kb1 Qb7+ Kc2 Qc7+ Kb3 Qb8+ Ka2"
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

move_white_memory = []
move_black_memory = []

white_next_moves = []
move_ratio = []
mode = 1

while not state.is_game_over():
        move_memory = False
        print(env.render())
        board_txt = state.fen().split()[0]
        board_encoded = ''.join(str(ord(c)) for c in board_txt)
        obs = make_matrix(state)
        print(str(obs))
        final_moves = []
        
        if cnt % 2 == 0:
            print(" ")
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
            print(" ")
            print("Black")
            player = 'black'
            #board_state = [np.reshape(obs, [1, 64]), 0, 0]
            board_state = list(np.reshape(obs, [1, 64])[0])
            board_state.append(0)
            board_state.append(opp)
            board_state = np.array(board_state)

        legal_moves = [state.san(x) for x in env.legal_moves]

        #Next Piece predict
        #X = board_state
        #max_val,act_values_pieces = model.predict(np.reshape(X, [1, 66]))
        #print("Next Piece: ",max_val,act_values_pieces )

        #Result predict
        #X = board_state
        #max_val,act_values = model_result.predict(np.reshape(X, [1, 66]))
        #print("Result: ",max_val,act_values)
        
        #act_values = list(act_values[0])
        #pieces_ordered = []
        #for i in act_values:
        #    max_val = max(act_values)
        #    pieces_ordered.append(act_values.index(max_val))
        #    act_values[act_values.index(max_val)] = -1
            
        #pieces_ordered = [pieces_dict[str(x)] for x in pieces_ordered]
        #pieces_ordered = pieces_ordered[:4]

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
        for mv in next_moves:
            final_moves.append(mv)

        #MinMaxTree
        #best_move = calculate_min_max_tree(state, env, player, depth=2, mode='stockfish', engine=engine)
        #if cnt % 2 == 0:
        #    best_move = calculate_min_max_tree2(state, env, player, depth=2, mode='stockfish', engine=engine,list_moves=next_moves, next_pieces=list(act_values_pieces[0]))
        #    print("MinMax Tree: ", best_move)

        #Player
        print(cnt)
        if cnt % 2 == 0:
            player = 'white'
            #df2 = df1[df1['white'].isin(['Kasparov, Garry','Kasparov, G.','Kasparov Garry (RUS)'])]
            df2 = df1[df1['black'].isin(['Carlsen, Magnus','Carlsen, M.','Carlsen,M'])]
            df2 = df2[df2['board'] == str(obs)]
            next_moves = []
            if len(df2) >0:
                move_memory = True
                move_white_memory.append(1)
                next_moves = list(set(list(df2['next_move'])))
                print('Next moves Player: ', next_moves)
                for mv in next_moves:
                    final_moves.append(mv)
            else:
                move_white_memory.append(0)
            '''if move_memory:
                #best_move = calculate_min_max_tree2(state, env, player, depth=0, mode='stockfish', engine=engine,list_moves=next_moves, next_pieces=list(act_values_pieces[0]))
                best_move = calculate_min_max_tree2(state, env, player, depth=0, mode='stockfish', engine=engine,list_moves=next_moves, next_pieces=None)
                next_move_memory = best_move
                print("Player: ", best_move)
                final_moves.append(best_move)'''
            
        else:
            '''player = 'black'
            #df2 = df1[df1['black'].isin(['Carlsen, Magnus','Carlsen, M.','Carlsen,M'])]
            df2 = df1[df1['white'].isin(['Kasparov, Garry','Kasparov, G.','Kasparov Garry (RUS)'])]
            #df2 = df1[df1['white'].isin(['Wesley So','So, W.','So, Wesley'])]
            df2 = df1
            df2 = df2[df2['board'] == str(obs)]
            next_moves = []
            if len(df2) >0:
                move_memory = True
                move_black_memory.append(1)
                next_moves = list(set(list(df2['next_move'])))
                print('Next moves Player: ', next_moves)
                for mv in next_moves:
                    final_moves.append(mv)
            else:
                move_black_memory.append(0)
            
            best_move2 = calculate_min_max_tree2(state, env, player, depth=1, mode='stockfish', engine=engine,list_moves=next_moves, next_pieces=list(act_values_pieces[0]))
            next_move_memory = best_move2
            print("Player: ", best_move2)
            final_moves.append(best_move2)'''
            final_moves.append(legal_moves[random.randrange(len(legal_moves))])
            
            
        
        #MCTS
        '''#root,selected_node,weights,material_adv = MCTS.calculate_MCTS(state = state, env = env,filter_prop='next_piece' ,filter_moves=pieces_ordered, par='material_adv')
        if len(white_next_moves) > 0 and player == 'white':
            print("Next moves memory: ", white_next_moves)
            if white_next_moves[0] in [state.san(x) for x in env.legal_moves]:
                best_move = white_next_moves[0]
                print("Next move memory: ", white_next_moves[0])
                white_next_moves = white_next_moves[2:]
            else:
                print("Generate next checkpoint plan..")
                white_next_moves = []
                          
        if len(df2) == 0 and player == 'white' and len(white_next_moves) == 0:
            if cnt % 2 == 0:
                #root,selected_node,weights,material_adv = MCTS.calculate_MCTS_checkpoint(state = state, env = env, memory = df1[(df1['result'] == '1-0')|(df1['result'] == '1/2-1/2')], engine=engine) #df1
                root,selected_node,weights,material_adv = MCTS.calculate_MCTS_checkpoint(state = state, env = env, memory = df1, engine=None)
            else:
                #root,selected_node,weights,material_adv = MCTS.calculate_MCTS_checkpoint(state = state, env = env, memory = df1[(df1['result'] == '0-1')|(df1['result'] == '1/2-1/2')], engine=engine)
                root,selected_node,weights,material_adv = MCTS.calculate_MCTS_checkpoint(state = state, env = env, memory = df1, engine=engine)
            print("MCTS: ",selected_node.parent_action)
            print("Next checkpoint on: ",len(selected_node.steps))
            best_move = selected_node.parent_action
            white_next_moves = selected_node.steps
            white_next_moves = white_next_moves[1:]
            #best_move2 = best_move
        #print(selected_node.material_adv)
        #print(material_adv)'''

        #Similarity
        if not move_memory:
            if cnt % 2 == 0:
                next_moves = get_moves_by_similarity(state,legal_moves)
                for mv in next_moves:
                    final_moves.append(mv)


        #get best final move
        a = 2
        if move_memory and a != 2:
            best_move = next_move_memory
        else:
            for mv in legal_moves:
                if '#' in mv:
                    final_moves.append(mv)
                elif '+' in mv:
                    final_moves.append(mv)
                elif 'x' in mv:
                    final_moves.append(mv)
            final_moves = list(set(final_moves))
            final_moves = [x for x in final_moves if x in legal_moves]
            print('Final Moves: ',final_moves)
            final_moves_scores = []
            for mv in final_moves:
                new_state = copy.deepcopy(state)
                action = new_state.push_san(mv)
                info = engine.analyse(new_state, chess.engine.Limit(time=0.1))
                mate = info["score"].white().mate()
                if mate != None:
                    if mate < 0:
                        score = -((20-(-mate)) * 1000)
                    else:
                        score = (20-mate) * 1000
                else:
                    score = info["score"].white().score()
                    
                score = score if score != None else 0
                final_moves_scores.append(score)
            
            best_move = final_moves[np.argmax(final_moves_scores)]

        print('--------------')
        print("Best Move Scores: ", final_moves_scores)
        print("Best Move Stacked: ", best_move)
        print("Legal Moves ", len(legal_moves), " Final moves ", len(final_moves), " Prop % ",100*(len(final_moves)/len(legal_moves)), " %")
        

        if cnt % 2 == 0:
            move = best_move
            move_ratio.append((len(final_moves)/len(legal_moves)))
        else:
            #legal_moves = [state.san(x) for x in env.legal_moves]
            #move = legal_moves[random.randrange(len(legal_moves))]
            #move = best_move2
            #move = best_move
            if mode == 1:
                result = engine.play(state, chess.engine.Limit(time=0.1)) #stock
                move = state.san(result.move)
            else:
                move = input("Insira um valor: ")
                
            print("Black final move: ", move)
            
        history_move.append(move)
        action = state.push_san(move)
        cnt +=1
        #take action
        try:
            state,reward,done,_ = env.step(action)
        except:
            break

plt.subplot(221)
plt.plot([i for i in range(len(move_white_memory))], move_white_memory)
plt.ylabel("moves")
plt.xlabel("move")
plt.title("move_white_memory")

plt.subplot(222)
plt.plot([i for i in range(len(move_ratio))], move_ratio)
plt.title("Move Ratio White - Final moves / Legal Moves")
plt.show()
