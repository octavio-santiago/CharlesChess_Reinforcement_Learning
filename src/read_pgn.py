"""
@author: Octavio Bomfim Santiago
Created on 10/07/2020
"""

import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import re
import pandas as pd
import pickle
from matplotlib import style
from collections import deque
import matplotlib.pyplot as plt
import itertools
import datetime as dt
import copy
import glob


import gym
import gym_chess
import random
import chess
import chess.engine
import chess.pgn
from gym_chess.alphazero import BoardEncoding

from difflib import SequenceMatcher
from chessRL import make_matrix

df1 = pd.read_pickle("../data/pgn/chess_deep_memory.pkl")
files = glob.glob("../data/pgn/*.pgn")

for f in files:
    print(f)
    #pgn = open("../data/pgn/master_games.pgn")
    pgn = open(f)

    X = []
    y = []

    games = []
    for i in range(0,10):
        f_game = chess.pgn.read_game(pgn)
        games.append(f_game)

    for first_game in games:
        #first_game = chess.pgn.read_game(game)
        print(first_game.headers["Event"])
        print(first_game.headers["Result"])
        result = first_game.headers["Result"]
        white = first_game.headers["White"]
        black = first_game.headers["Black"]
        try:
            white_rating = first_game.headers["WhiteElo"]
        except:
            white_rating = 0
        try:
            black_rating = first_game.headers["BlackElo"]
        except:
            black_rating = 0
        board = first_game.board()
        
        # Iterate through all moves and play them on a board.
        for move in first_game.mainline_moves():
            if first_game.is_end():
                break
            obs = make_matrix(board)
            X.append(obs)
            y.append(board.san(move))
            #print(board.san(move))
            board.push(move)

    df = pd.DataFrame({'board':X, 'next_move':y})
    df.board = df.board.astype(str)
    df['result'] = result
    df['white'] = white
    df['black'] = black
    df['white_rating'] = white_rating
    df['black_rating'] = black_rating

    df1 = df1.append(df)


df1 = df1.drop_duplicates()
df1.to_pickle("../data/pgn/chess_deep_memory.pkl")
#X = np.array(X)
#y = np.array(y)

#np.save("../data/pgn/chess_deep_memory_X_train", X)
#np.save("../data/pgn/chess_deep_memory_y_train", y)
    
