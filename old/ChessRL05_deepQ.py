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

path = r'C:\Users\Octavio\Desktop\Projetos Python\Chess-RL\games.csv'
df = pd.read_csv(path)
df = df.dropna(subset=['moves'])
df = df[(df['winner'] == 'white') & (df['victory_status'] == 'mate')].reset_index()

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.999    # discount rate #0.95
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9 #0.99
        self.learning_rate = 1e-5 #0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(5, input_dim=self.state_size, activation='relu')) #24
        model.add(Dense(5, activation='relu')) #24
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate),
                      metrics=['accuracy'])
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        ##With probability e select a random action at
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        ##otherwise select at = maxa Q∗(φ(st), a; θ)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size): # train
        ##Sample random minibatch of transitions (φj , aj , rj , φj+1) from D
        minibatch = random.sample(self.memory, batch_size)
        #print(minibatch[0])
        ##Set yj = rj for terminal φj+1 or (rj + γ maxa0 Q(φj+1, a0; θ)) for non-terminal φj+1
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                #a = self.target_model.predict(next_state)[0]
                max_future_q = np.amax(a)
                target[0][action] = reward + self.gamma * max_future_q
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
                
            ##Perform a gradient descent step on (yj − Q(φj , aj ; θ))^2  
            history = self.model.fit(state, target, epochs=5, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return history

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
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

#load
t_len = 10
model_name = f"chessML_len{t_len}.h5"
token_name = f"tokenizer_len{t_len}.pkl"

with open(token_name, 'rb') as handle:
    tokenizer = pickle.load(handle)

stats = pd.read_csv('train_data_log.csv')
vocabulary_size = stats['vocabulary_size'][0]
seq_len = stats['seq_len'][0]

model = create_model(vocabulary_size, seq_len)    
model.load_weights(model_name)



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

#predict(seq_len,model, tokenizer)

def make_matrix(board): #type(board) == chess.Board()
    pgn = board.epd()
    foo = []  #Final board
    pieces = pgn.split(" ", 1)[0]
    rows = pieces.split("/")

    mapped = {
        'P': 1,     # White Pawn
        'p': -1,    # Black Pawn
        'N': 2,     # White Knight
        'n': -2,    # Black Knight
        'B': 3,     # White Bishop
        'b': -3,    # Black Bishop
        'R': 4,     # White Rook
        'r': -4,    # Black Rook
        'Q': 5,     # White Queen
        'q': -5,    # Black Queen
        'K': 6,     # White King
        'k': -6     # Black King
        }
    
    for row in rows:
        foo2 = []  #This is the row I make
        for thing in row:
            if thing.isdigit():
                for i in range(0, int(thing)):
                    foo2.append(0)
            else:
                foo2.append(mapped[thing])
        foo.append(foo2)
    return foo

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def act(cnt,df,seq_len, model, tokenizer, obs, legal_moves, opening: str=None, idx_open: int=0, debug = False, choice = 2):
    #start
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
                    return move
                else:
                    for idx,line in enumerate(df[df['opening_name'] == opening]['moves']):
                        op_idx = int(df[df['opening_name'] == opening]['opening_ply'].iloc[idx])
                        moves = line.split()[cnt:]
                        if len(moves) >= 3:
                            move = moves[2]
                            #print("teste: ", move)
                            if move in legal_moves:
                                print("teste certo: ", move)
                                return move
                        else:
                            return legal_moves[0]
   
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
                            print("Next move: ",moves[moves.index(obs[-1]) + 1])
                            print("game: ",moves[moves.index(obs[-1])-2:moves.index(obs[-1])+2])
                        move = moves[moves.index(obs[-1]) + 1]
                        if move in legal_moves:
                            break
                    except:
                        #ML predict
                        #print("Machine Learning")
                        board = ' '.join([str(elem) for elem in obs])
                        next_moves = predict(seq_len,model, tokenizer, board)
                        move = next_moves
                        move = [x.capitalize() if len(x) > 2 else x for x in move]
                        for i in move:
                            if i in legal_moves:
                                move = i
                                break
                else:
                    #ML predict
                    #print("Machine Learning")
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
            result = engine.play(state, chess.engine.Limit(time=0.1))
            move = state.san(result.move)
            return move
                    
                            
episode_rewards = []
epsilon_time = []
moving_avg = []
action_time = []
choices_time = []
EPISODES = 64
SHOW_EVERY = 2

if __name__ == "__main__":
    
    env = gym.make('Chess-v0')
    print(env.render())

    state_size = 64
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    try:
        agent.load("chess-ddqn.h5")
    except:
        pass

    batch_size = 64 #256
    move =""

    for e in range(EPISODES):
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
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
                    
                    move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves,opp,choice)
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
                    move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves,choice=2)
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
                    
                    move = act(cnt,df,seq_len, model,tokenizer,obs,legal_moves,opp,choice)
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
            agent.save("chess-ddqn.h5")
        
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
    plt.plot([i for i in range(len(choices_time))], choices_time)
    plt.title("Choices")

    plt.show()



