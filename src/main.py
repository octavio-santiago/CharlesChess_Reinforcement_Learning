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

from difflib import SequenceMatcher

from chessRL import make_matrix
from chessRL import DQNAgent
from chessRL import similar
from chessRL import act


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



if __name__ == "__main__":
    #load matches dataset
    path = '../data/games.csv'
    df = pd.read_csv(path)
    df = df.dropna(subset=['moves'])
    df = df[(df['winner'] == 'white') & (df['victory_status'] == 'mate')].reset_index()

    analytics_df = pd.DataFrame()

    #load LSTM model
    t_len = 10
    model_name = f"../models/lstm_model/chessML_len{t_len}.h5"
    token_name = f"../models/lstm_model/tokenizer_len{t_len}.pkl"

    with open(token_name, 'rb') as handle:
            tokenizer = pickle.load(handle)

    stats = pd.read_csv('../logs/train_data_log.csv')
    vocabulary_size = int(stats['vocabulary_size'][0])
    seq_len = int(stats['seq_len'][0])

    model = create_model(vocabulary_size, seq_len)    
    model.load_weights(model_name)

    #train parameters
    episode_rewards = []
    epsilon_time = []
    moving_avg = []
    action_time = []
    choices_time = []
    EPISODES = 64
    SHOW_EVERY = 2

    env = gym.make('Chess-v0')
    print(env.render())

    state_size = 64
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("../models/dqn_agent/chess-ddqn.h5")
    except:
        pass

    batch_size = 64 #256
    move =""

    for e in range(EPISODES):
        
        engine = chess.engine.SimpleEngine.popen_uci("../stockfish")
        engine1 = chess.engine.SimpleEngine.popen_uci("../komodo-12.1.1-64bit")
        engine2 = chess.engine.SimpleEngine.popen_uci("../Houdini_15a_x64")
        
        epsilon_time.append(agent.epsilon)
        if e % SHOW_EVERY == 0:
                print(f"on # {e}, epsilon: {agent.epsilon}")
                print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")

        episode_reward = 0
        scores_eps = []
        done = False
        state = env.reset()
        cnt = 0
        obs = []
        mode = "for two"
        
        history_moves = []

        '''string = 'e4 d5 exd5 Qxd5'
        game = string.split(' ')
        for move in game:
            action = state.push_san(move)          
            #take action
            state,reward,done,_ = env.step(action)
            cnt +=1'''
        
        while not done:
            
            board_txt = state.fen().split()[0]
            board_encoded = ''.join(str(ord(c)) for c in board_txt)
            #obs = [board_encoded]
            obs = make_matrix(state)
            #obs = BoardEncoding(env, history_length=4).action_space
            
            #print(done)
            #print(state.is_game_over())
            if state.is_checkmate():
                result = "White" if reward > 0 else "Black"
                print(f"CHECKMATE! for {result}")
                done = True
                break
            elif state.is_stalemate() or state.is_insufficient_material() or state.is_game_over() or state.can_claim_threefold_repetition() or state.can_claim_fifty_moves() or state.can_claim_draw() or state.is_fivefold_repetition() or state.is_seventyfive_moves():
                done = True
                break
            #elif "#" in move:
            #    done = True
                
            if done:
                    print(env.render(mode='unicode'))
                    agent.update_target_model()
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, reward, agent.epsilon))
                    print(f"episode: {e}, reward: {episode_reward}")
                    break

            #if e == EPISODES-1:
            #    action_time.append(action)
                
            if mode == 'alone':#########                    
                l = list(df[df['opening_eco'] == 'B50']['opening_ply'])[0]
                if cnt <= l:
                        #oppening
                        print("Opening")
                        m = list(df[df['opening_eco'] == 'B50']['moves'])[0]
                        m = str(m).split(' ')
                        #result = engine.play(state, chess.engine.Limit(time=0.1))
                        
                        #move = state.san(m[cnt])
                        move = m[cnt]
                        history_moves.append(move)
                        action = state.push_san(move)
                
                        #take action
                        state,reward,done,_ = env.step(action)
                        
                if cnt % 2 == 0 and cnt >l:
                    if cnt <= -1:
                        #oppening
                        print("Opening")
                        result = engine.play(state, chess.engine.Limit(time=0.1))
                        move = state.san(result.move)
                        history_moves.append(move)
                        action = state.push_san(move)
                
                        #take action
                        state,reward,done,_ = env.step(action)

                    else:
                        print("PC turn")
                        legal_moves = [state.san(x) for x in env.legal_moves]
                        
                        # Analysis #https://www.chessprogramming.org/Evaluation
                        # mobility
                        #state.pseudo_legal_moves
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
                        
                        #choose action
                        opp = 'Scandinavian Defense: Mieses-Kotroc Variation' 
                        if cnt == 2:
                            if move == 'd5':
                                opp = 'Scandinavian Defense: Mieses-Kotroc Variation' #e4 d5
                            elif move == 'e5':
                                opp = 'Scotch Game' #e4 e5
                            elif move == 'c6':
                                opp = 'Caro-Kann Defense' #e4 c6
                            elif move == 'c5':
                                opp = 'Sicilian Defense' #e4 c5
                            elif move == 'e6':
                                opp = 'French Defense: Knight Variation' #e4 e6
                                
                        choice = agent.act(np.reshape(obs, [1, 64]))
                        print(agent.model.predict(np.reshape(obs, [1, 64])))
                        choices_time.append(choice)
                        
                        #many engine
                        if choice == 0:
                            move = act(cnt,df,seq_len, model,tokenizer,history_moves,legal_moves,engine,state,opp,0,choice=choice)
                        elif choice == 1:
                            move = act(cnt,df,seq_len, model,tokenizer,history_moves,legal_moves,engine1,state,opp,0,choice=choice)
                        elif choice == 2:
                            move = act(cnt,df,seq_len, model,tokenizer,history_moves,legal_moves,engine2,state,opp,0,choice=choice)
                        
                        old_obs = obs.copy()
                        action = state.push_san(move)
                        
                        next_state,reward,done,_ = env.step(action)

                        info = engine.analyse(state, chess.engine.Limit(time=0.1))
                        mate = info["score"].white().mate()
                        if mate != None:
                            if mate < 0:
                                score = -((20-(-mate)) * 1000)
                            else:
                                score = (20-mate) * 1000
                        else:
                            score = info["score"].white().score()
                            
                        score = score if score != None else 0
                        print("Action: ",choice ," Score:", score, " Mate in: ", mate)
                        
                        scores_eps.append(score)
                        reward = score if score != None else 0
                        #reward = np.average(scores_eps)
                        
                        board_txt = next_state.fen().split()[0]
                        board_encoded = ''.join(str(ord(c)) for c in board_txt)
                        #obs = [board_encoded]
                        obs = make_matrix(state)
                        
                        agent.remember(np.reshape(old_obs, [1, 64]), choice, reward, np.reshape(obs, [1, 64]), done)
                        state = next_state
                        
                        episode_reward = reward

                        #append action       
                        history_moves.append(move)

                        #create Analytics df
                        analytics = pd.DataFrame({'move':[cnt], 'score':[score], 'mobility':[mob],
                                                  'material_adv':[material_adv], 'nega_max':[f],
                                                  'space':[space], 'center_control':[center_control]
                                                  })
                        analytics_df = analytics_df.append(analytics)
                        
                        ##Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
                        if len(agent.memory) > batch_size:
                            #print("train")
                            ##Set yj = rj for terminal φj+1 or (rj + γ maxa0 Q(φj+1, a0; θ)) for non-terminal φj+1
                            history = agent.replay(batch_size)
                            print(history.history['acc'])
                        
                        print(f"PC Move {move}")
                elif cnt % 2 != 0 and cnt >l:
                    print("Human turn")
                    valid = False
                    while not valid:
                        #move = input("Choose a move: ")
                        result = engine.play(state, chess.engine.Limit(time=0.1))
                        move = state.san(result.move)
                        history_moves.append(move)
                        try:
                            action = state.push_san(move)
                            valid = True
                        except:
                            valid = False
                            history_moves.pop()
                            print("Wrong move, choosing other move")
                    #take action
                    state,reward,done,_ = env.step(action)
                
                
                
                
            elif mode == "for two": ##########
                if cnt % 2 == 0:
                    print("PC turn")
                    legal_moves = [state.san(x) for x in env.legal_moves]
                    
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
                    
                    #choose action
                    opp = 'Scandinavian Defense: Mieses-Kotroc Variation' 
                    if cnt == 2:
                        if move == 'd5':
                            opp = 'Scandinavian Defense: Mieses-Kotroc Variation' #e4 d5
                        elif move == 'e5':
                            opp = 'Scotch Game' #e4 e5
                        elif move == 'c6':
                            opp = 'Caro-Kann Defense' #e4 c6
                        elif move == 'c5':
                            opp = 'Sicilian Defense' #e4 c5
                        elif move == 'e6':
                            opp = 'French Defense: Knight Variation' #e4 e6
                            
                    choice = agent.act(np.reshape(obs, [1, 64]))
                    print(agent.model.predict(np.reshape(obs, [1, 64])))
                    choices_time.append(choice)
                    
                    #many engine
                    if choice == 0:
                        move = act(cnt,df,seq_len, model,tokenizer,history_moves,legal_moves,engine,state,opp,0,choice=choice)
                    elif choice == 1:
                        move = act(cnt,df,seq_len, model,tokenizer,history_moves,legal_moves,engine1,state,opp,0,choice=choice)
                    elif choice == 2:
                        move = act(cnt,df,seq_len, model,tokenizer,history_moves,legal_moves,engine2,state,opp,0,choice=choice)
                        
                        
                    old_obs = obs.copy()
                    action = state.push_san(move)
                    
                    next_state,reward,done,_ = env.step(action)

                    info = engine.analyse(state, chess.engine.Limit(time=0.1))
                    mate = info["score"].white().mate()
                    if mate != None:
                        if mate < 0:
                            score = -((20-(-mate)) * 1000)
                        else:
                            score = (20-mate) * 1000
                    else:
                        score = info["score"].white().score()
                        
                    score = score if score != None else 0
                    print("Action: ",choice ," Score:", score, " Mate in: ", mate)
                    
                    scores_eps.append(score)
                    reward = score if score != None else 0
                    #reward = np.average(scores_eps)
                    
                    board_txt = next_state.fen().split()[0]
                    board_encoded = ''.join(str(ord(c)) for c in board_txt)
                    #obs = [board_encoded]
                    obs = make_matrix(state)
                    
                    agent.remember(np.reshape(old_obs, [1, 64]), choice, reward, np.reshape(obs, [1, 64]), done)
                    state = next_state
                    
                    episode_reward = reward

                    #append action       
                    history_moves.append(move)

                    #create Analytics df
                    analytics = pd.DataFrame({'move':[cnt], 'score':[score], 'mobility':[mob],
                                              'material_adv':[material_adv], 'nega_max':[f],
                                              'space':[space], 'center_control':[center_control]
                                              })
                    analytics_df = analytics_df.append(analytics)
                    
                    ##Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
                    if len(agent.memory) > batch_size:
                        #print("train")
                        ##Set yj = rj for terminal φj+1 or (rj + γ maxa0 Q(φj+1, a0; θ)) for non-terminal φj+1
                        history = agent.replay(batch_size)
                        print(history.history['acc'])
                    
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
                       
                

            #print(env.render(mode='unicode'))
            print(state.is_game_over())
            if state.is_game_over():
                done = True
                break
            cnt += 1
            #print(" ")

        # if e % 10 == 0:
        if e % 2 == 0:
            agent.save("../models/dqn_agent/chess-ddqn.h5")
        
        env.close()
        engine.quit()
        engine1.quit()
        engine2.quit()
        episode_rewards.append(episode_reward)
        
    moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")
    plt.subplot(221)
    #plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
    plt.plot([i for i in range(len(moving_avg))], moving_avg)
    plt.ylabel(f"reward {SHOW_EVERY}ma")
    plt.xlabel("episode #")
    plt.title("Episode rewards Mov Avg")

    plt.subplot(222)
    plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
    #plt.plot([i for i in range(len(epsilon_time))], epsilon_time)
    plt.title("Episode rewards")

    plt.subplot(223)
    plt.plot([i for i in range(len(epsilon_time))], epsilon_time)
    plt.title("Epsilon")

    plt.subplot(224)
    plt.plot([i for i in range(len(choices_time))], choices_time) #
    plt.title("Choices")

    plt.show()
