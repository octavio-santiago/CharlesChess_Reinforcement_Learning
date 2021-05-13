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
            #return 3 #lerning with stockfish
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
     
