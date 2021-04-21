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


if __name__ == "__main__":
    #load matches dataset
    path = '../data/games.csv'
    df = pd.read_csv(path)
    df = df.dropna(subset=['moves'])
    df = df[(df['winner'] == 'white') & (df['victory_status'] == 'mate')].reset_index()

    #load LSTM model
    t_len = 10
    model_name = f"../models/lstm_model/chessML_len{t_len}.h5"
    token_name = f"../models/lstm_model/tokenizer_len{t_len}.pkl"

    with open(token_name, 'rb') as handle:
            tokenizer = pickle.load(handle)

    stats = pd.read_csv('../logs/train_data_log.csv')
    vocabulary_size = stats['vocabulary_size'][0]
    seq_len = stats['seq_len'][0]

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
        
        while not done:
            
            board_txt = state.fen().split()[0]
            board_encoded = ''.join(str(ord(c)) for c in board_txt)
            #obs = [board_encoded]
            obs = make_matrix(state)
            #obs = BoardEncoding(env, history_length=4).action_space
            
            print(done)
            print(state.is_game_over())
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

            if e == EPISODES-1:
                action_time.append(action)
                
            if mode == 'alone':
                legal_moves = [state.san(x) for x in env.legal_moves]
                #choose action
                if cnt % 2 == 0: #white
                    #print("White turn")
                    opp = 'Scandinavian Defense: Mieses-Kotroc Variation' 
                    if cnt == 2:
                        if move == 'd5':
                            opp = 'Scandinavian Defense: Mieses-Kotroc Variation' #e4 d5
                        elif move == 'e5':
                            opp = 'Scotch Game' #e4 e5
                            
                    choice = agent.act(np.reshape(obs, [1, 64]))
                    print(agent.model.predict(np.reshape(obs, [1, 64])))
                    choices_time.append(choice)
                    
                    move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves,engine,state,opp,choice)
                    old_obs = obs.copy()
                    action = state.push_san(move)
                    
                    next_state,reward,done,_ = env.step(action)

                    info = engine.analyse(state, chess.engine.Limit(time=0.1))
                    mate = info["score"].white().mate()
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
                    
                else: #black
                    #print("Black turn")
                    move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves,engine,state,choice=2)
                    action = state.push_san(move)
                    #take action
                    state,reward,done,_ = env.step(action)

                #append action       
                history_moves.append(move)
                
                ##Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
                if len(agent.memory) > batch_size:
                    #print("train")
                    ##Set yj = rj for terminal φj+1 or (rj + γ maxa0 Q(φj+1, a0; θ)) for non-terminal φj+1
                    history = agent.replay(batch_size)
                    print(history.history['acc'])
                
                
                
                
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
                            
                    choice = agent.act(np.reshape(obs, [1, 64]))
                    print(agent.model.predict(np.reshape(obs, [1, 64])))
                    choices_time.append(choice)
                    
                    move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves,engine,state,opp,choice)
                    old_obs = obs.copy()
                    action = state.push_san(move)
                    
                    next_state,reward,done,_ = env.step(action)

                    info = engine.analyse(state, chess.engine.Limit(time=0.1))
                    mate = info["score"].white().mate()
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
            cnt += 1
            #print(" ")

        # if e % 10 == 0:
        if e % 2 == 0:
            agent.save("../models/dqn_agent/chess-ddqn.h5")
        
        env.close()
        engine.quit()
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
