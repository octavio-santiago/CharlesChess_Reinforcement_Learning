import itertools
import numpy as np

from chessRL import make_matrix
import copy

import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding

def Evaluate(player,state):
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
    
    if player == 'white':
        return material_adv
    else:
        return - material_adv

    return material_adv
    
def Search(depth, player, env, state):
    if depth == 0:
        #print(env.render())
        return Evaluate(player,state)

    # end of the game (checkmate)
    legal_moves = [state.san(x) for x in env.legal_moves]
    moves = legal_moves
    if state.is_game_over(): #reduces the moves
        if state.result() == '1-0' and player == 'white':
            return 10000
        elif state.result() == '1-0' and player == 'black':
            return -10000
        if state.result() == '0-1' and player == 'white':
            return -10000
        elif state.result() == '0-1' and player == 'black':
            return 10000
        elif state.result() == '1/2-1/2':
            return 0  

    best_evaluation = -10000

    for move in moves:
        #board make a move
        state_new = copy.deepcopy(state)
        env_new = copy.deepcopy(env)
        action = state_new.push_san(move)
        state_new,reward,done,_ = env_new.step(action)
        evaluation = - Search(depth -1, player, env_new, state_new)
        best_evaluation = max(evaluation, best_evaluation)
        #board unmake move

    return best_evaluation

def AlphaSearch(depth, player, env, state, alpha, beta, evaluations, max_depth):
    if depth == 0:
        evaluation = Evaluate(player,state)
        #evaluations.append(evaluation)
        #evaluations.append(lines)
        #print(env.render())
        return evaluation

    # end of the game (checkmate)
    if state.is_game_over(): #reduces the moves
        #print("Checkmate!")
        if state.result() == '1-0' and player == 'white':
            return 10000
        elif state.result() == '1-0' and player == 'black':
            return -10000
        if state.result() == '0-1' and player == 'white':
            return -10000
        elif state.result() == '0-1' and player == 'black':
            return 10000
        elif state.result() == '1/2-1/2':
            return 0  

    legal_moves = [state.san(x) for x in env.legal_moves]
    '''#Move Ordering
    if player == 'white':
        opp = False
    else:
        opp = True
    #checkmate
    moves = [x for x in legal_moves if '#' in x]
    #check
    moves2 = [x for x in legal_moves if '+' in x]
    for m in moves2:
        moves.append(m)
        
    #capture
    for i in [5,4,3,2,1]: #1=pawn
        #find the oppenent piece
        squares = state.pieces(i,opp)
        square_list = [chess.square_name(square) for square in squares]
        #for square in squares:
        #    square_list.append(square)    
        #chess.square_name(21)
        #see if we are attacking it
        for j in square_list:
            capt = [x for x in legal_moves if 'x'+str(j) in x]
            for k in capt:
                moves.append(k)
    
    #moves = legal_moves
    myset = set(moves)
    moves = list(myset)'''
    moves = legal_moves
    
    #print(legal_moves)
    #print(moves)
    
    #best_evaluation = -10000
    for move in moves:
        #if depth == max_depth:
            #evaluations.append("*")
            #evaluations.append(move)
        #evaluations.append(move)
        #lines.append(move)
        #board make a move
        state_new = copy.deepcopy(state)
        env_new = copy.deepcopy(env)
        action = state_new.push_san(move)
        state_new,reward,done,_ = env_new.step(action)
        evaluation = - AlphaSearch(depth -1, player, env_new, state_new, -beta, -alpha, evaluations, depth)
        if evaluation >= beta:
            return beta
        alpha = max(alpha,evaluation)
        #best_evaluation = max(evaluation, best_evaluation)
        #board unmake move

    return alpha
    

'''evaluations = []
env = gym.make('Chess-v0')
state = env.reset()

string = "e4 c5 Nc3 Nc6 Qf3 e5"
game = string.split(' ')
for move in game:
    #move = state.san(action)
    action = state.push_san(move)
                
    #take action
    state,reward,done,_ = env.step(action)


legal_moves = [state.san(x) for x in env.legal_moves]
evals = []
for move in legal_moves:
    state_new = copy.deepcopy(state)
    env_new = copy.deepcopy(env)
    action = state_new.push_san(move)
    state_new,reward,done,_ = env_new.step(action)
    #best_move = Search(depth=5, player ='white', env=env, state=state)
    best_move = AlphaSearch(depth=4, player ='white', env=env_new, state=state_new,alpha=-10000, beta=10000,
                        evaluations=evaluations, max_depth=4)
    evals.append(best_move)
    
print(evals, legal_moves)'''
#str(evaluations).split('*')[1].replace(' ','').split(',')


def calculate_min_max_tree(state, env, player, depth):
    evaluations = []
    legal_moves = [state.san(x) for x in env.legal_moves]
    evals = []

    #Move Ordering
    if player == 'white':
        opp = False
    else:
        opp = True
    #checkmate
    moves = [x for x in legal_moves if '#' in x]
    #check
    moves2 = [x for x in legal_moves if '+' in x]
    for m in moves2:
        moves.append(m)
        
    #capture
    for i in [5,4,3,2,1]: #1=pawn
        #find the oppenent piece
        squares = state.pieces(i,opp)
        square_list = [chess.square_name(square) for square in squares]
        #for square in squares:
        #    square_list.append(square)    
        #chess.square_name(21)
        #see if we are attacking it
        for j in square_list:
            capt = [x for x in legal_moves if 'x'+str(j) in x]
            for k in capt:
                moves.append(k)
    
    #moves = legal_moves
    myset = set(moves)
    legal_moves = list(myset)
    if len(legal_moves) == 0:
        legal_moves = [state.san(x) for x in env.legal_moves]
    
    for move in legal_moves:
        state_new = copy.deepcopy(state)
        env_new = copy.deepcopy(env)
        action = state_new.push_san(move)
        state_new,reward,done,_ = env_new.step(action)
        #best_move = Search(depth=5, player ='white', env=env, state=state)
        best_eval = AlphaSearch(depth=depth, player ='white', env=env_new, state=state_new,alpha=-10000, beta=10000,
                            evaluations=evaluations, max_depth=depth)
        if player == 'black':
            best_eval = -best_eval
        evals.append(best_eval)
        
    print(evals, legal_moves)
    best_move = legal_moves[evals.index(max(evals))]
    return best_move
