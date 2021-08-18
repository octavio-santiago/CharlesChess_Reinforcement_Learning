import numpy as np
from collections import defaultdict
import pickle
import pandas as pd
import random
import matplotlib.pyplot as plt

#import MCTS
#from min_max_tree import calculate_min_max_tree, calculate_min_max_tree2

######

#from keras.utils import np_utils

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
#from keras.layers import Embedding
#from keras.preprocessing.sequence import pad_sequences
#from keras.optimizers import Adam
#from keras import backend as K
#from sklearn.preprocessing import LabelEncoder

####
import gym
import gym_chess
import random
import chess
#import chess.engine
#from gym_chess.alphazero import BoardEncoding

from chessRL import make_matrix
import copy

from scipy.spatial.distance import cdist


def get_moves_by_similarity(state, legal_moves):
    df = pd.read_excel(r'C:\Users\Octavio\Desktop\Projetos Python\Chess-RL\data\pgn\chess_deep_memory_excel.xlsx')

    matr = df.loc[:,'s1':]

    #env = gym.make('Chess-v0')
    #state = env.reset()

    #cnt = 0

    #string = "e4 c5 Nc3 Nc6 Qf3 e5"
    #string = "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 Re1 Nd6 Nxe5 Be7 Bf1 Nxe5 Rxe5 O-O d4 Bf6 Re1 Re8 Bf4 Rxe1"
    #game = string.split(' ')
    #for move in game:
    #    #move = state.san(action)
    #    action = state.push_san(move)
    #                
    #    #take action
    #    state,reward,done,_ = env.step(action)
    #    cnt +=1

    #print(env.render())
    #print(state.status())

    board_txt = state.fen().split()[0]
    board_encoded = ''.join(str(ord(c)) for c in board_txt)
    obs = make_matrix(state)
    board_state = [list(np.reshape(obs, [1, 64])[0])]
    print("Board state: ", board_state)

    moves = []
    #legal_moves = [state.san(x) for x in env.legal_moves] ##

    #position = [0,0,0,0,0,0,0,0]

    print('COSINE DISTANCE')
    cos_dist = cdist(board_state, matr[:], metric='cosine')
    #print(cos_dist)
    #print(df.iloc[np.argmax(cos_dist),:].next_move)

    df['cos_sim'] = cos_dist[0]
    df = df.sort_values('cos_sim', ascending=False)
    top3 = list(set(list(df[['cos_sim']].iloc[:,:].cos_sim)))[-1:]
    print(top3)
    #n_moves = list(df[['next_move']].iloc[:20,:].next_move)
    #n_moves = [x for x in n_moves if x in legal_moves]
    df2 = df[df['cos_sim'].isin(top3)]
    n_moves = list(df2[['next_move']].iloc[:,:].next_move)
    print(len(n_moves))
    
    for mv in n_moves:
        moves.append(mv)
    #print(n_moves)
        
    #print(df.head(10))

    print(" ")
    print('EUCLIDEAN DISTANCE')
    euc_dist = cdist(board_state, matr[:], metric='euclidean')
    #print(euc_dist)
    #print(df.iloc[np.argmax(euc_dist),:].next_move)

    df['euc_sim'] = euc_dist[0]
    df = df.sort_values('euc_sim', ascending=False)
    #print(df.head(10))
    #n_moves = list(df[['next_move']].iloc[:20,:].next_move)
    #n_moves = [x for x in n_moves if x in legal_moves]
    top3 = list(set(list(df[['euc_sim']].iloc[:,:].euc_sim)))[-1:]
    print(top3)
    df2 = df[df['euc_sim'].isin(top3)]
    n_moves = list(df2[['next_move']].iloc[:,:].next_move)
    print(len(n_moves))
    #print(n_moves)
    for mv in n_moves:
        moves.append(mv)

    print(" ")
    print('jaccard DISTANCE')
    dist = cdist(board_state, matr[:], metric='jaccard')
    #print(dist)
    #print(df.iloc[np.argmax(dist),:].next_move)

    df['jacc_sim'] = dist[0]
    df = df.sort_values('jacc_sim', ascending=False)
    #print(df.head(10))
    #n_moves = list(df[['next_move']].iloc[:20,:].next_move)
    #n_moves = [x for x in n_moves if x in legal_moves]
    top3 = list(set(list(df[['jacc_sim']].iloc[:,:].jacc_sim)))[-1:]
    print(top3)
    df2 = df[df['jacc_sim'].isin(top3)]
    n_moves = list(df2[['next_move']].iloc[:,:].next_move)
    print(len(n_moves))
    #print(n_moves)
    for mv in n_moves:
        moves.append(mv)

    print(" ")
    print('correlation DISTANCE')
    dist = cdist(board_state, matr[:], metric='correlation')
    #print(dist)
    #print(df.iloc[np.argmax(dist),:].next_move)

    df['corr_sim'] = dist[0]
    df = df.sort_values('corr_sim', ascending=False)
    #print(df.head(10))
    #n_moves = list(df[['next_move']].iloc[:20,:].next_move)
    #n_moves = [x for x in n_moves if x in legal_moves]
    top3 = list(set(list(df[['corr_sim']].iloc[:,:].corr_sim)))[-1:]
    print(top3)
    df2 = df[df['corr_sim'].isin(top3)]
    n_moves = list(df2[['next_move']].iloc[:,:].next_move)
    print(len(n_moves))
    #print(n_moves)
    for mv in n_moves:
        moves.append(mv)

    print(" ")
    print('chebyshev DISTANCE')
    dist = cdist(board_state, matr[:], metric='chebyshev')
    #print(dist)
    #print(df.iloc[np.argmax(dist),:].next_move)

    df['cheb_sim'] = dist[0]
    df = df.sort_values('cheb_sim', ascending=False)
    #print(df.head(10))
    #n_moves = list(df[['next_move']].iloc[:20,:].next_move)
    #n_moves = [x for x in n_moves if x in legal_moves]
    top3 = list(set(list(df[['cheb_sim']].iloc[:,:].cheb_sim)))[-1:]
    print(top3)
    df2 = df[df['cheb_sim'].isin(top3)]
    n_moves = list(df2[['next_move']].iloc[:,:].next_move)
    print(len(n_moves))
    #print(n_moves)
    for mv in n_moves:
        moves.append(mv)

    '''print(" ")
    print("King Position")
    king_now = board_state[0].index(-6)

    def get_king_pos(row, king_now):
        list_pieces = list(row['s1':'s64'])
        idx = list_pieces.index(-6)
        if idx == king_now:
            return 2
        elif idx >= (king_now-1) and idx <= (king_now+1):
            return 1
        else:
            return 0
        
    df['king_pos'] = df.apply(lambda row: get_king_pos(row, king_now) , axis=1)
    df = df.sort_values('king_pos', ascending=False)
    #print(df.head(10))
    #n_moves = list(df[['next_move']].iloc[:20,:].next_move)
    #n_moves = [x for x in n_moves if x in legal_moves]
    top3 = [2]
    df2 = df[df['king_pos'].isin(top3)]
    n_moves = list(df2[['next_move']].iloc[:,:].next_move)
    print(len(n_moves))
    #print(n_moves)
    for mv in n_moves:
        moves.append(mv)'''

    moves = list(set(moves))
    moves = [x for x in moves if x in legal_moves]
    return moves
