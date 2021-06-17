import itertools
import numpy as np
import pandas as pd

from chessRL import make_matrix
import copy

import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding

def Evaluate_min_max(player,state):
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

def Evaluate_space(player, state):
    board = make_matrix(state)
    #Space https://en.wikipedia.org/wiki/Chess_strategy#Space
    #attacked
    att_space = 0
    #ocupied
    if player == 'white':
        black_space = list(itertools.chain(*board))[:32]
        occ = [x if x>0 else 0 for x in black_space]
    else:
        white_space = list(itertools.chain(*board))[33:]
        occ = [x if x<0 else 0 for x in white_space]
        
    #space
    space = np.sum(occ) + att_space
    #print("Space: ", space)
    return space


def Evaluate_mobility(player,env, state):
    try:
        legal_moves = [state.san(x) for x in env.legal_moves]
    except:
        return 100
    pseudo_legal_moves = [state.san(x) for x in state.pseudo_legal_moves]                  
    # Analysis #https://www.chessprogramming.org/Evaluation
    # mobility
    #state.pseudo_legal_moves
    mob = len(legal_moves)
    #print('Mobility: ', mob)
    
    return mob

def Evaluate_center_control(player,env, state):
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
    board = make_matrix(state)
    #center control
    space_list = list(itertools.chain(*board))
    center_list = [space_list[27], space_list[28], space_list[35], space_list[36]]
    center_list_val = [mapped_val[str(x)] for x in center_list]
    center_control = np.sum(center_list_val)
    #print("Center Control: ", center_control) 
    
    return center_control

def Evaluate_development(player,env, state):
    board = make_matrix(state)
    #Space https://en.wikipedia.org/wiki/Chess_strategy#Space
    #attacked
    att_space = 0
    #ocupied
    if player == 'white':
        white_space = list(itertools.chain(*board))[33:57] 
        occ = [x if x>0 else 0 for x in white_space]
        occ = [x if x<6 else 0 for x in occ] # remove king
    else:
        black_space = list(itertools.chain(*board))[8:32]
        occ = [x if x<0 else 0 for x in black_space]
        occ = [x if x>-6 else 0 for x in occ] # remove king
        
    #space
    space = abs(np.sum(occ)) + att_space
    #print("Space: ", space)
    return space

def Evaluate_space_def(player,env, state):
    score_vals = []
    attack_piec = {'p':1,'n':20,'b':20,'r':40,'q':80, 'k':1,
                   'P':1,'N':20,'B':20,'R':40,'Q':80, 'K':1}
    att_w = [0,0,50,75,88,94,97,99]
    if player == 'white':
        opp = True
        opp1 = False
    else:
        opp = False
        opp1 = True
    for i in [2,3,4,5,6]:
        b = [square for square in state.pieces(i,opp)] #square
        for j in b:
            s = [square for square in state.attackers(opp1,int(j))] #attackers
            p = [state.piece_at(square) for square in state.attackers(opp1,int(j))] #black attackers to white king
            p = [attack_piec[str(x)] for x in p] #black attackers to white king
            
            attackers_score = -(np.sum(p) * len(s))/100
            score_vals.append(attackers_score)
            
    return np.sum(score_vals)
    
def Evaluate_king_safety(player,env, state):
    #We multpily the number of attacked squares by a constant: 20 for a knight, 20 for a bishop, 40 for a rook and 80 for a queen. The result of multiplication is added to valueOfAttacks. After finding all attacks, we look at attackingPiecesCount, use it as an index to the table given below, and our king attack score is valueOfAttacks * attackWeight[attackingPiecesCount] / 100.
    #https://www.chessprogramming.org/King_Safety
    safety_index = [
       0, 0,   0,   1,   1,   2,   3,   4,   5,   6,
       8,  10,  13,  16,  20,  25,  30,  36,  42,  48,
      55,  62,  70,  80,  90, 100, 110, 120, 130, 140,
     150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
     250, 260, 270, 280, 290, 300, 310, 320, 330, 340,
     350, 360, 370, 380, 390, 400, 410, 420, 430, 440,
     450, 460, 470, 480, 490, 500, 510, 520, 530, 540,
     550, 560, 570, 580, 590, 600, 610, 620, 630, 640,
     650, 650, 650, 650, 650, 650, 650, 650, 650, 650,
     650, 650, 650, 650, 650, 650, 650, 650, 650, 650
    ]
    attack_piec = {'p':1,'n':20,'b':20,'r':40,'q':80, 'k':1,
                   'P':1,'N':20,'B':20,'R':40,'Q':80, 'K':1}
    att_w = [0,0,50,75,88,94,97,99]
    score_vals = []
    if player == 'white':
        opp = True
        opp1 = False
    else:
        opp = False
        opp1 = True
    #casteling
    #legal_moves = [state.san(x) for x in env.legal_moves]
    #if 'O-O' or 'O-O-O' in legal_moves:
    b = [square for square in state.pieces(6,True)] #square king
    b_list = [b[0], b[0] +1, b[0]-1, b[0]+8, b[0]-8, b[0]+7,b[0]-7,b[0]+9, b[0]-9]
    b_list = [ x for x in b_list if (x>=0 and x<=63)]
    for j in b_list:
        s = [square for square in state.attackers(opp1,int(j))] #attackers
        p = [state.piece_at(square) for square in state.attackers(opp1,int(j))] #black attackers to white king
        p = [attack_piec[str(x)] for x in p] #black attackers to white king
        attackers_score = -(np.sum(p) * len(s))/100
        score_vals.append(attackers_score)
        
    
    #s = [square for square in state.attackers(False,int(b[0]))] #black attackers to white king
    #p = [state.piece_at(square) for square in state.attackers(False,int(b[0]))] #black attackers to white king
    #p = [attack_piec[x] for x in p] #black attackers to white king
    
    #safety_score = safety_index[b[0]]
    #attackers_score = -(np.sum(p) * len(s))/100
    return np.sum(score_vals)

    
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

def AlphaSearch(depth, player, env, state, alpha, beta, evaluations, max_depth, mode, engine=None):
    if depth == 0:
        if mode == 'material_adv':
            evaluation = Evaluate_min_max(player,state)
        elif mode =='space':
            evaluation = Evaluate_space(player, state)
        elif mode == 'mobility':
            evaluation = Evaluate_mobility(player,env, state)
        elif mode == 'center_control':
            evaluation = Evaluate_center_control(player,env, state)
        elif mode == 'development':
            evaluation = Evaluate_development(player,env, state)
        elif mode == 'space_def':
            evaluation = Evaluate_space_def(player,env, state)
        elif mode == 'king_safety':
            evaluation = Evaluate_king_safety(player,env, state)
        elif mode == 'stockfish':
            evaluation = engine.analyse(state, chess.engine.Limit(time=0.1))
            mate = evaluation["score"].white().mate()
            evaluation = evaluation["score"].white().score()
            if mate != None:
                evaluation = (20-mate) * 1000
            evaluation = evaluation
        #evaluations.append(evaluation)
        #evaluations.append(lines)
        #print(env.render())
        #print(evaluation)
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
    #TODO: Ordering Moves
    
    #print(legal_moves)
    #print(moves)
    
    #best_evaluation = -10000
    for move in moves:
        #print(move)
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
        evaluation = AlphaSearch(depth -1, player, env_new, state_new, -beta, -alpha, evaluations, depth, mode, engine)
        if depth % 2 == 0:
            evaluation = - evaluation
            
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


def calculate_min_max_tree(state, env, player, depth, mode, engine=None):
    evaluations = []
    legal_moves = [state.san(x) for x in env.legal_moves]
    evals = []
    scores = []

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
    legal_moves = [state.san(x) for x in env.legal_moves]
    for move in legal_moves:
        state_new = copy.deepcopy(state)
        env_new = copy.deepcopy(env)
        action = state_new.push_san(move)
        state_new,reward,done,_ = env_new.step(action)
        info = engine.analyse(state_new, chess.engine.Limit(time=0.1))
        mate = info["score"].white().mate()
        if mate != None:
            if mate < 0:
                score = -((20-(-mate)) * 1000)
            else:
                score = (20-mate) * 1000
        else:
            score = info["score"].white().score()
        
        
        scores.append(score)        
        #best_move = Search(depth=5, player ='white', env=env, state=state)
        best_eval = - AlphaSearch(depth=depth, player ='white', env=env_new, state=state_new,alpha=-10000, beta=10000,
                            evaluations=evaluations, max_depth=depth, mode=mode, engine=engine)
        if depth == 0:
            best_eval = - best_eval
        if player == 'black':
            best_eval = -best_eval
        evals.append(best_eval)
        
    print(evals, legal_moves)
    df = pd.DataFrame({'legal_moves':legal_moves, 'evals':evals, 'scores':scores})
    df = df.sort_values('evals',ascending=False)
    max_val = df.iloc[0,1]
    if len(df) > 5 :
        if len(df[df['evals'] == max_val]) > 5:
            df = df[df['evals'] == max_val]
        else:   
            df = df.iloc[:,:]
    df = df.sort_values('scores',ascending=False)
    print(df)
    #best_move = legal_moves[evals.index(max(evals))]
    best_move = df.iloc[0,0]
    return best_move


def calculate_min_max_tree2(state, env, player, depth, mode, engine=None, list_moves=None, next_pieces=None):
    evaluations = []
    evals = []
    legal_moves = [state.san(x) for x in env.legal_moves]
    list_moves = [x for x in list_moves if x in legal_moves]
    move_orig = copy.deepcopy(list_moves)
    moves = list_moves
    
    #checkmate
    moves2 = [x for x in legal_moves if '#' in x]
    #check
    moves3 = [x for x in legal_moves if '+' in x]
    for m in moves2:
        moves.append(m)
    for m in moves3:
        moves.append(m)

    #capture
    if player == 'white':
        opp = False
    else:
        opp = True
    for i in [5,4,3,2,1]: #1=pawn
        #find the oppenent piece
        squares = state.pieces(i,opp)
        square_list = [chess.square_name(square) for square in squares]
        #see if we are attacking it
        for j in square_list:
            capt = [x for x in legal_moves if 'x'+str(j) in x]
            for k in capt:
                moves.append(k)
        
    while len(moves) < 10:
        try:
            if next_pieces.index(max(next_pieces)) == 0:
                lm = [x for x in legal_moves if len(x) <=2]
                moves.append(lm[random.randint(0,len(lm)-1)])
            elif next_pieces.index(max(next_pieces)) == 1 :
                lm = [x for x in legal_moves if 'N' in x]
                moves.append(lm[random.randint(0,len(lm)-1)])
            elif next_pieces.index(max(next_pieces)) == 2 :
                lm = [x for x in legal_moves if 'B' in x]
                moves.append(lm[random.randint(0,len(lm)-1)])
            elif next_pieces.index(max(next_pieces)) == 3 :
                lm = [x for x in legal_moves if 'R' in x]
                moves.append(lm[random.randint(0,len(lm)-1)])
            elif next_pieces.index(max(next_pieces)) == 4 :
                lm = [x for x in legal_moves if 'Q' in x]
                moves.append(lm[random.randint(0,len(lm)-1)])
            elif next_pieces.index(max(next_pieces)) == 5 :
                lm = [x for x in legal_moves if 'K' in x]
                moves.append(lm[random.randint(0,len(lm)-1)])
            else:        
                moves.append(legal_moves[random.randint(0,len(legal_moves)-1)])
        except:
            moves.append(legal_moves[random.randint(0,len(legal_moves)-1)])
            
        
    #moves = legal_moves
    myset = set(moves)
    legal_moves = list(myset)

    if mode == 'stockfish':
        legal_moves = move_orig
    if len(legal_moves) == 0:
        legal_moves = [state.san(x) for x in env.legal_moves]
    
    for move in legal_moves:
        print(move)
        state_new = copy.deepcopy(state)
        env_new = copy.deepcopy(env)
        action = state_new.push_san(move)
        state_new,reward,done,_ = env_new.step(action)
        #best_move = Search(depth=5, player ='white', env=env, state=state)
        best_eval = AlphaSearch(depth=depth, player = player, env=env_new, state=state_new,alpha=-10000, beta=10000,
                            evaluations=evaluations, max_depth=depth, mode=mode, engine=engine)
        #if player == 'black':
        #    best_eval = -best_eval
        evals.append(best_eval)
        
    print(evals, legal_moves)
    if depth % 2 == 0:
        meval= evals.index(max(evals))
        print(meval)
        best_move = legal_moves[meval]
    else:
        meval= evals.index(min(evals))
        print(meval)
        best_move = legal_moves[meval]
    return best_move
