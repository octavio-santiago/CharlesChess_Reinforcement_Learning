"""
@author: Octavio Bomfim Santiago
Created on 10/07/2020
"""

from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from keras.utils import to_categorical
import pandas as pd
import pickle
from matplotlib import style
from collections import deque
import matplotlib.pyplot as plt
import itertools

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding

from chessRL import make_matrix

from difflib import SequenceMatcher

mapped_val = {
            '0':0,
            '1': 1,     # White Pawn
            '-1': -1,    # Black Pawn
            '2': 3,     # White Knight
            '-2': -3,    # Black Knight
            '3': 3,     # White Bishop
            '-3': -3,    # Black Bishop
            '4': 5,     # White Rook
            '-4': -5,    # Black Rook
            '5': 9,     # White Queen
            '-5': -9,    # Black Queen
            '6': 0,     # White King
            '-6': 0     # Black King
            }

def predict(seq_len,model, tokenizer, board, debug = False):
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

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def act(cnt,df,seq_len, model, tokenizer, obs, legal_moves, engine, state, opening: str=None, idx_open: int=0, debug = False, choice = 2):
    #start
    print("Choice: ",choice)
    if cnt == 0: 
        if opening == None:
            #random opening
            idx = random.choice([x for x in range(0,len(df['moves']))])
            moves = df['moves'][idx].split()
            move = moves[0]
        else:
            moves = df[df['opening_name'] == opening]['moves'].iloc[idx_open].split()
            move = moves[0]
            
        return move

    else:
        
        if choice == 0:  
            if opening != None:
                if (cnt <= int(df[df['opening_name'] == opening]['opening_ply'].iloc[idx_open])):
                    moves = df[df['opening_name'] == opening]['moves'].iloc[idx_open].split()
                    move = moves[cnt]
                    print("Option 0 opening")
                    return move
                else:
                    print("Option 0 Material Adv")
                    result = engine.play(state, chess.engine.Limit(time=0.1))
                    move = state.san(result.move)

                    legal_moves_scores = []
                    geral_scores = []
                    for move in legal_moves:
                        action = state.push_san(move) # Make the move
                        obs = make_matrix(state)
                        #material adv
                        pieces_adv = [mapped_val[str(x)] for x in list(itertools.chain(*obs))]
                        material_adv = np.sum(pieces_adv)
                        #print('Material Adv: ', material_adv)
                        legal_moves_scores.append(material_adv)

                        info = engine.analyse(state, chess.engine.Limit(time=0.1))
                        mate = info["score"].white().mate()
                        if mate != None:
                            if mate < 0:
                                score = -((20-(-mate)) * 100)
                            else:
                                score = (20-mate) * 100
                        else:
                            score = info["score"].white().score()
                            
                        score = score if score != None else 0
                        geral_scores.append(score)

                        state.pop()  # Unmake the last move
                        
                    #do the best move
                    #print(legal_moves_scores)
                    a = min(legal_moves_scores)
                    b = max(legal_moves_scores)
                    c = min(geral_scores)
                    d = max(geral_scores)
                    n_legal_score = [(x-a)/(b-a) if (b-a) != 0 else 0 for x in legal_moves_scores]
                    n_geral_score = [(x-c)/(d-c) if (d-c) != 0 else 0 for x in geral_scores]
                    zipped_lists = zip(n_legal_score, n_geral_score)
                    final_score = [x + y for (x, y) in zipped_lists]
                    #move = legal_moves[legal_moves_scores.index(max(legal_moves_scores))]
                    move = legal_moves[final_score.index(max(final_score))]
                    return move

   
        elif choice == 1:
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
                            print("Option 1 Similarity")
                            print("Next move: ",moves[moves.index(obs[-1]) + 1])
                            print("game: ",moves[moves.index(obs[-1])-2:moves.index(obs[-1])+2])
                            
                        move = moves[moves.index(obs[-1]) + 1]
                        if move in legal_moves:
                            return move
                            #break
                    except:
                        #ML predict
                        print("Option 1 Machine Learning")
                        board = ' '.join([str(elem) for elem in obs])
                        next_moves = predict(seq_len,model, tokenizer, board)
                        move = next_moves
                        move = [x.capitalize() if len(x) > 2 else x for x in move]
                        for i in move:
                            if i in legal_moves:
                                move = i
                                return move
                                #break
                else:
                    #ML predict
                    #print("Machine Learning")
                    print("Option 1 Machine Learning")
                    board = ' '.join([str(elem) for elem in obs])
                    next_moves = predict(seq_len,model, tokenizer, board)
                    move = next_moves
                    move = [x.capitalize() if len(x) > 2 else x for x in move]
                    for i in move:
                        if i in legal_moves:
                            move = i
                            return move
                        
                    move = random.choice(legal_moves)
                    return move
                        

        elif choice == 2:
            if opening != None:
                if (cnt <= int(df[df['opening_name'] == opening]['opening_ply'].iloc[idx_open])):
                    moves = df[df['opening_name'] == opening]['moves'].iloc[idx_open].split()
                    move = moves[cnt]
                    print("Option 0 opening")
                    return move
                else:
                    print("Option 2 Space")
                    result = engine.play(state, chess.engine.Limit(time=0.1))
                    move = state.san(result.move)

                    legal_moves_scores = []
                    geral_scores = []
                    for move in legal_moves:
                        action = state.push_san(move) # Make the move
                        obs = make_matrix(state)
                        #Space https://en.wikipedia.org/wiki/Chess_strategy#Space
                        #attacked
                        att_space = 0
                        #ocupied
                        black_space = list(itertools.chain(*obs))[:32]
                        occ = [x if x>0 else 0 for x in black_space]
                        #space
                        space = np.sum(occ) + att_space
                        #print("Space: ", space)
                        legal_moves_scores.append(space)

                        info = engine.analyse(state, chess.engine.Limit(time=0.1))
                        mate = info["score"].white().mate()
                        if mate != None:
                            if mate < 0:
                                score = -((20-(-mate)) * 100)
                            else:
                                score = (20-mate) * 100
                        else:
                            score = info["score"].white().score()
                            
                        score = score if score != None else 0
                        geral_scores.append(score)

                        state.pop()  # Unmake the last move
                        
                    #do the best move
                    #print(legal_moves_scores)
                    a = min(legal_moves_scores)
                    b = max(legal_moves_scores)
                    c = min(geral_scores)
                    d = max(geral_scores)
                    n_legal_score = [(x-a)/(b-a) if (b-a) != 0 else 0 for x in legal_moves_scores]
                    n_geral_score = [(x-c)/(d-c) if (d-c) != 0 else 0 for x in geral_scores]
                    zipped_lists = zip(n_legal_score, n_geral_score)
                    final_score = [x + y for (x, y) in zipped_lists]

                    print("Final score: ",final_score)
                    print("Legal moves: ",legal_moves)
                    move = legal_moves[final_score.index(max(final_score))]
                    
                    #move = legal_moves[legal_moves_scores.index(max(legal_moves_scores))]
                    return move

        elif choice == 3: #stockfish backup
            print("Option 3 - Exploration Stockfish")
            result = engine.play(state, chess.engine.Limit(time=0.1))
            move = state.san(result.move)
            return move
            
            
