# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:31:39 2021

@author: Octavio
"""

import gym
import gym_chess
import random
import chess
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import style
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf


style.use("ggplot")


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.7  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
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
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(env.legal_moves)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                #t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



# run
episode_rewards = []
epsilon_time = []
moving_avg = []
action_time = []
EPISODES = 40
SHOW_EVERY = 2

if __name__ == "__main__":
    env = gym.make('Chess-v0')
    print(env.render())
    #games = env.games
    #state_size = env.state_size
    #action_size = env.action_size
    state_size = 1
    action_size = 1
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    batch_size = 64

    for e in range(EPISODES):
        epsilon_time.append(agent.epsilon)
        if e % SHOW_EVERY == 0:
                print(f"on # {e}, epsilon: {agent.epsilon}")
                print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")

        episode_reward = 0
        done = False
        state = env.reset()
        #state = np.reshape(state, [1, state_size])
        board_txt = state.fen().split()[0]
        board_encoded = ''.join(str(ord(c)) for c in board_txt)
        obs = [board_encoded]
        history = []
        
        while not done:
            if state.is_checkmate() or state.is_stalemate() or state.is_insufficient_material() or state.is_game_over() or state.can_claim_threefold_repetition() or state.can_claim_fifty_moves() or state.can_claim_draw() or state.is_fivefold_repetition() or state.is_seventyfive_moves():
                done = True
            #env.render()
            action = agent.act(np.reshape(obs, [1, len(obs)]))
            #action = random.choice(env.legal_moves)
            
            if e == EPISODES-1:
                action_time.append(action)
                
            #next_state, reward, done, lucro0,odd, lucro = env.step(action,time_h,lucro0) #action in env
            next_state,reward,done,_ = env.step(action)
            reward = (reward*1000) - 1
            #next_state = np.reshape(next_state, [1, state_size])
            #agent.remember(state, action, reward, next_state, done)
            old_obs = obs.copy()
            history.append(state.san(action))
            agent.remember(np.reshape(old_obs, [1, len(old_obs)]), action, reward, np.reshape(obs, [1, len(obs)]), done)
            state = next_state
            
            episode_reward += reward
            
            #print(f"lucro: {lucro0}, reward: {reward}")
                
            if done:
                print(env.render(mode='unicode'))
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, reward, agent.epsilon))
                print(f"episode: {e}, reward: {episode_reward}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        episode_rewards.append(episode_reward)
            
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")
        
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode = "valid")
plt.subplot(211)
#plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
plt.plot([i for i in range(len(epsilon_time))], epsilon_time)
#plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel("episode #")

plt.subplot(212)
plt.plot([i for i in range(len(episode_rewards))], episode_rewards)
#plt.plot([i for i in range(len(carteira_time))], carteira_time)
#plt.plot([i for i in range(len(lucro_time))], lucro_time)

plt.show()



'''env = gym.make('Chess-v0')
print(env.render())

state = env.reset()
done = False
#print(state)

#print(env.legal_moves)
#move = state.push_san('d4')
#env.step(move)
#print(env.render(mode='unicode'))


#move = chess.Move.from_uci('e7e5')
#env.step(move)
#print(env.render(mode='unicode'))

#state.is_checkmate()

while not done:
    if state.is_checkmate() or state.is_stalemate() or state.is_insufficient_material() or state.is_game_over() or state.can_claim_threefold_repetition() or state.can_claim_fifty_moves() or state.can_claim_draw() or state.is_fivefold_repetition() or state.is_seventyfive_moves():
        done = True
    action = random.choice(env.legal_moves)
    state,reward,done,_ = env.step(action)
    print(env.render(mode='unicode'))
    
    print(" ")

env.close()'''
