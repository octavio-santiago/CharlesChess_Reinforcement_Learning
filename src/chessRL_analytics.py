import itertools
import numpy as np

import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding

from chessRL import make_matrix


def calculate_mobility(env, state):
    legal_moves = [state.san(x) for x in env.legal_moves]
                        
    # Analysis #https://www.chessprogramming.org/Evaluation
    # mobility
    #state.pseudo_legal_moves
    mob = len(legal_moves) + len(state.pseudo_legal_moves)
    print('Mobility: ', mob)

def calculate_material_adv(state):
    board = make_matrix(state)
    # material_adv
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
    pieces_adv = [mapped_val[str(x)] for x in list(itertools.chain(*board))]
    material_adv = np.sum(pieces_adv)
    #print('Material Adv: ', material_adv)
    return material_adv

def calculate_nega_max():  
    #NegaMax
    wK = list(itertools.chain(*obs)).count(6)
    bK = list(itertools.chain(*obs)).count(-6)
    wQ = list(itertools.chain(*obs)).count(5)
    bQ = list(itertools.chain(*obs)).count(-5)
    wR = list(itertools.chain(*obs)).count(4)
    bR = list(itertools.chain(*obs)).count(-4)
    wB = list(itertools.chain(*obs)).count(3)
    bB = list(itertools.chain(*obs)).count(-3)
    wN = list(itertools.chain(*obs)).count(2)
    bN = list(itertools.chain(*obs)).count(-2)
    wP = list(itertools.chain(*obs)).count(1)
    bP = list(itertools.chain(*obs)).count(-1)
    f = 200*(wK-bK) + 9*(wQ-bQ) + 5*(wR-bR)+ 3*(wB-bB + wN-bN) + 1*(wP-bP)
    print("NegaMax: ", f)
    
def calculate_space(state):
    board = make_matrix(state)
    #Space https://en.wikipedia.org/wiki/Chess_strategy#Space
    #attacked
    att_space = 0
    #ocupied
    black_space = list(itertools.chain(*board))[:32]
    occ = [x if x>0 else 0 for x in black_space]
    #space
    space = np.sum(occ) + att_space
    print("Space: ", space)
    return space

def calculate_center_control(state):
    board = make_matrix(state)
    #center control
    space_list = list(itertools.chain(*board))
    center_list = [space_list[27], space_list[28], space_list[35], space_list[36]]
    center_list_val = [mapped_val[str(x)] for x in center_list]
    center_control = np.sum(center_list_val)
    print("Center Control: ", center_control) 
