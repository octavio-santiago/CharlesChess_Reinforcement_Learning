"""
@author: Octavio Bomfim Santiago
Created on 10/07/2020
"""

from keras.preprocessing.text import Tokenizer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
from tensorflow.keras.utils import to_categorical
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
from tensorflow.keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding

from min_max_tree import calculate_min_max_tree

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

def act(cnt,df,seq_len, model, tokenizer, obs, legal_moves, engine, state, opening: str=None, idx_open: int=0, debug = False, choice = 2, env = None):
    #start
    print("Choice: ",choice)
    #TODO: adapt two players
    player = 'white'
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
            print("Option 0 - Material Advantage")
            move = calculate_min_max_tree(state, env, player, depth=0, mode='material_adv', engine=engine)
            #move = state.san(result.move)
            return move

        elif choice == 1:
            print("Option 1 - Space")
            move = calculate_min_max_tree(state, env, player, depth=0, mode='space', engine=engine)
            #move = state.san(result.move)
            return move
        
        elif choice == 2:
            print("Option 2 - Mobility")
            move = calculate_min_max_tree(state, env, player, depth=0, mode='mobility', engine=engine)
            #move = state.san(result.move)
            return move
        
        elif choice == 3:
            print("Option 3 - center_control")
            move = calculate_min_max_tree(state, env, player, depth=0, mode='center_control', engine=engine)
            #move = state.san(result.move)
            return move
        
        elif choice == 4:
            print("Option 4 - development")
            move = calculate_min_max_tree(state, env, player, depth=0, mode='development', engine=engine)
            #move = state.san(result.move)
            return move
        
        elif choice == 5:
            print("Option 5 - space_def")
            move = calculate_min_max_tree(state, env, player, depth=0, mode='space_def', engine=engine)
            #move = state.san(result.move)
            return move
        elif choice == 6:
            print("Option 6 - king_safety")
            move = calculate_min_max_tree(state, env, player, depth=0, mode='king_safety', engine=engine)
            #move = state.san(result.move)
            return move
            
            
