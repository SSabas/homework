"""
Project:
    UC Berkeley CS294 Deep Reinforcement Learning Homework I
Description:
    Supervised deep learning models for imitation learning
Authors:
  Sven Sabas
Date:
  25/06/2019
"""

# ----------------------------- IMPORT LIBRARIES -------------------------------- #

import argparse
import os
import pickle
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import importlib
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD

import helpers

# ------------------------ BUILD KERAS NN MODEL --------------------------------- #

class SupervisedModel():

    def __init__(self, env):
        input_len, output_len = env.observation_space.shape[0], env.action_space.shape[0]
        self.model = Sequential()
        self.model.add(Dense(units=64, input_dim=input_len, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(units=output_len))
        self.model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

    def train(self, data, epochs, verbose=True, val_prop=0.1):
        X_train, X_val, y_train, y_val = train_test_split(data['observations'],
                                                        np.squeeze(data['actions']),
                                                        test_size=val_prop)
        self.model.fit(X_train, y_train,
                       batch_size=128,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=(X_val, y_val))

    def save(self, filename):
        self.model.save_weights(filename)

    def load(self, filename):
        self.model.load_weights(filename)

    def plot(self, filename):
        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.getcwd() + '/results/' + filename + '.pdf')

    def save_results(self, filename):
        with open(os.getcwd() + '/results/' + filename + '.pkl', "wb") as f:
            pickle.dump(self.model.history.history, f)



def evaluate_model
