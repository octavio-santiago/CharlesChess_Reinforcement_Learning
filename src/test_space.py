import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding
import itertools
import numpy as np
import re

from chessRL import make_matrix

env = gym.make('Chess-v0')
print(env.render())
engine = chess.engine.SimpleEngine.popen_uci("../stockfish")

scores_eps = []
done = False
state = env.reset()

cnt = 0
while not done:

    board_txt = state.fen().split()[0]
    board_encoded = ''.join(str(ord(c)) for c in board_txt)
    obs = make_matrix(state)
    if cnt == 0:
        move = "e4"
        action = state.push_san(move)
        state,reward,done,_ = env.step(action)
        
    elif cnt % 2 == 0:
        legal_moves = [state.san(x) for x in env.legal_moves]
        legal_moves_scores = []

        for move in legal_moves:
            action = state.push_san(move) # Make the move
            obs = make_matrix(state)

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
            
            state.pop()  # Unmake the last move

        #do the best move
        print(legal_moves_scores)
        move = legal_moves[legal_moves_scores.index(max(legal_moves_scores))]
        action = state.push_san(move)
        state,reward,done,_ = env.step(action)

        
    else:
        print("Human turn")
        valid = False
        while not valid:
            #move = input("Choose a move: ")
            result = engine.play(state, chess.engine.Limit(time=0.1))
            move = state.san(result.move)
            #obs.append(move)
            try:
                action = state.push_san(move)
                valid = True
            except:
                valid = False
                #obs.pop()
                print("Wrong move, choosing other move")
        #take action
        state,reward,done,_ = env.step(action)
        


            
    print(done)
    print(state.is_game_over())
    if state.is_game_over():
        done = True
        break

    cnt += 1
    
env.close()
engine.quit()





        
        
'''                
    # Analysis #https://www.chessprogramming.org/Evaluation
    # mobility
    mob = len(legal_moves)
    print('Mobility: ', mob)
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
    pieces_adv = [mapped_val[str(x)] for x in list(itertools.chain(*obs))]
    material_adv = np.sum(pieces_adv)
    print('Material Adv: ', material_adv)
    
    #Space https://en.wikipedia.org/wiki/Chess_strategy#Space
    #attacked
    att_space = 0
    #ocupied
    black_space = list(itertools.chain(*obs))[:32]
    occ = [x if x>0 else 0 for x in black_space]
    #space
    space = np.sum(occ) + att_space
    print("Space: ", space)

    #center control
    space_list = list(itertools.chain(*obs))
    center_list = [space_list[27], space_list[28], space_list[35], space_list[36]]
    center_list_val = [mapped_val[str(x)] for x in center_list]
    center_control = np.sum(center_list_val)
    print("Center Control: ", center_control)
    
    old_obs = obs.copy()
    action = state.push_san(move)

    next_state,reward,done,_ = env.step(action)

    info = engine.analyse(state, chess.engine.Limit(time=0.1))
    mate = info["score"].white().mate()
    if mate != None:
        score = (20-mate) * 10000
    else:
        score = info["score"].white().score()
        
    score = score if score != None else 0
    print("Action: ",choice ," Score:", score, " Mate in: ", mate)
    
    scores_eps.append(score)
    #reward = score if score != None else 0
    reward = np.average(scores_eps)
    
    board_txt = next_state.fen().split()[0]
    board_encoded = ''.join(str(ord(c)) for c in board_txt)
    #obs = [board_encoded]
    obs = make_matrix(state)
    
    agent.remember(np.reshape(old_obs, [1, 64]), choice, reward, np.reshape(obs, [1, 64]), done)
    state = next_state
    
    episode_reward = reward

    #append action       
    history_moves.append(move)

    print(done)
    print(state.is_game_over())
    if state.is_game_over():
        done = True
        break
    env.close()
    engine.quit()'''
