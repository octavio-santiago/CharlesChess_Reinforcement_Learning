from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from tensorflow.keras.utils import to_categorical
import pandas as pd
import pickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences

import gym
import gym_chess
import random
import chess

from difflib import SequenceMatcher

#https://python-chess.readthedocs.io/en/latest/

path = r'C:\Users\Octavio\Desktop\Projetos Python\Chess-RL\games.csv'
df = pd.read_csv(path)
df = df.dropna(subset=['moves'])
df = df[(df['winner'] == 'white') & (df['victory_status'] == 'mate')].reset_index()

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

#load
t_len = 10
model_name = f"chessML_len{t_len}.h5"
token_name = f"tokenizer_len{t_len}.pkl"

with open(token_name, 'rb') as handle:
    tokenizer = pickle.load(handle)

stats = pd.read_csv('train_data_log.csv')
vocabulary_size = stats['vocabulary_size'][0]
seq_len = stats['seq_len'][0]

model = create_model(vocabulary_size, seq_len)    
model.load_weights(model_name)



def predict(seq_len,model, tokenizer,board, debug = False):
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

#predict(seq_len,model, tokenizer)
def predict_enemy(obs,state):
    new_obs = obs.copy()
    #predict my move
    board = ' '.join([str(elem) for elem in obs])
    next_moves = predict(seq_len,model, tokenizer, board)
    move = next_moves
    move = [x.capitalize() if len(x) > 2 else x for x in move]
    for i in move:
        if i in legal_moves:
            move = i
            break
    new_obs.append(move)
    state.push_san(move)
    legal_moves = [state.san(x) for x in state.legal_moves]
    
    #predict enemy's move
    board = ' '.join([str(elem) for elem in new_obs])
    next_moves = predict(seq_len,model, tokenizer, board)
    move = next_moves
    move = [x.capitalize() if len(x) > 2 else x for x in move]
    for i in move:
        if i in legal_moves:
            move = i
            break
    new_obs.append(move)
    state.push_san(move)
    legal_moves = [state.san(x) for x in state.legal_moves]
    state.pop()
    state.pop()
    #compare to memory
    
    
    #calculate next steps
    
    



def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def act(cnt,df,seq_len, model, tokenizer, obs, legal_moves, opening: str=None, idx_open: int=0, debug = False):
    #start
    if cnt == 0:
        if opening == None:
            #random opening
            idx = random.choice([x for x in range(0,len(df['moves']))])
            moves = df['moves'][idx].split()
            move = moves[0]
        else:
            moves = df[df['opening_name'] == opening]['moves'].iloc[idx_open].split()
            move = moves[0]

    else:
        if opening != None:
            if (cnt <= int(df[df['opening_name'] == opening]['opening_ply'].iloc[idx_open])):
                moves = df[df['opening_name'] == opening]['moves'].iloc[idx_open].split()
                move = moves[cnt]
                return move
            else:
                for idx,line in enumerate(df[df['opening_name'] == opening]['moves']):
                    op_idx = int(df[df['opening_name'] == opening]['opening_ply'].iloc[idx])
                    moves = line.split()[cnt:]
                    if len(moves) >= 3:
                        move = moves[2]
                        print("teste: ", move)
                        if move in legal_moves:
                            print("teste certo: ", move)
                            return move
   
        df2 = df.copy()
        df2['similarities'] = df2['moves'].apply(lambda x: similar(x.split(), obs))
        df2 = df2.sort_values(by=['similarities'], ascending = False)
        #moves = moves[cnt-1:]
        for i in range(0,len(df2['moves'])):
            moves = df2['moves'][i].split()
            if debug:
                print("Similarity of , ", df2['similarities'][i] * 100, "%")
                print(obs[-1])
            if obs[-1] in moves:
                try:
                    if debug:
                        #similarity
                        print("Next move: ",moves[moves.index(obs[-1]) + 1])
                        print("game: ",moves[moves.index(obs[-1])-2:moves.index(obs[-1])+2])
                    move = moves[moves.index(obs[-1]) + 1]
                    if move in legal_moves:
                        break
                except:
                    #ML predict
                    print("Machine Learning")
                    board = ' '.join([str(elem) for elem in obs])
                    next_moves = predict(seq_len,model, tokenizer, board)
                    move = next_moves
                    move = [x.capitalize() if len(x) > 2 else x for x in move]
                    for i in move:
                        if i in legal_moves:
                            move = i
                            break
            else:
                #ML predict
                print("Machine Learning")
                board = ' '.join([str(elem) for elem in obs])
                next_moves = predict(seq_len,model, tokenizer, board)
                move = next_moves
                move = [x.capitalize() if len(x) > 2 else x for x in move]
                for i in move:
                    if i in legal_moves:
                        move = i
                        break
                
                        
    return move
    


env = gym.make('Chess-v0')
print(env.render())

done = False
state = env.reset()
cnt = 0
obs = []
mode = "for two"
while not done:
    if state.is_checkmate():
        result = "White" if reward > 0 else "Black"
        print(f"CHECKMATE! for {result}")
        done = True
    elif state.is_stalemate() or state.is_insufficient_material() or state.is_game_over() or state.can_claim_threefold_repetition() or state.can_claim_fifty_moves() or state.can_claim_draw() or state.is_fivefold_repetition() or state.is_seventyfive_moves():
        done = True

    if mode == 'alone':
        legal_moves = [state.san(x) for x in env.legal_moves]
        #choose action
        if cnt % 2 == 0: #white
            print("White turn")
            opp = 'Scandinavian Defense: Mieses-Kotroc Variation'
            move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves,opp)
        else: #black
            print("Black turn")
            move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves)
        #append action
        #obs.append(state.san(action))        
        obs.append(move)
        try:
            action = state.push_san(move)
        except:
            obs.pop()
            print("Random")
            move = state.san(random.choice(env.legal_moves))
            obs.append(move)
            action = state.push_san(move)   
        #take action
        state,reward,done,_ = env.step(action)
        
    elif mode == "for two":
        if cnt % 2 == 0:
            print("PC turn")
            legal_moves = [state.san(x) for x in env.legal_moves]
            #choose action
            opp = 'Scandinavian Defense: Mieses-Kotroc Variation' 
            if cnt == 2:
                if move == 'd5':
                    opp = 'Scandinavian Defense: Mieses-Kotroc Variation' #e4 d5
                elif move == 'e5':
                    opp = 'Scotch Game' #e4 e5
            move = act(cnt,df,seq_len, model,tokenizer,obs, legal_moves, opp)
            #append action
            #obs.append(state.san(action))
            obs.append(move)
            try:
                action = state.push_san(move)
            except:
                obs.pop()
                print("Random")
                move = state.san(random.choice(env.legal_moves))
                obs.append(move)
                action = state.push_san(move)
            #take action
            state,reward,done,_ = env.step(action)
            print(f"PC Move {move}")
        else:
            print("Human turn")
            valid = False
            while not valid:
                move = input("Choose a move: ")
                obs.append(move)
                try:
                    action = state.push_san(move)
                    valid = True
                except:
                    valid = False
                    obs.pop()
                    print("Wrong move, choosing other move")
            #take action
            state,reward,done,_ = env.step(action)
               
        

    print(env.render(mode='unicode'))
    cnt += 1
    print(" ")


env.close()
