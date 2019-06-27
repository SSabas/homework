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

import os
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split

# ------------------------ BUILD KERAS NN MODEL --------------------------------- #


class SupervisedModel():

    def __init__(self, env, layers=(16, 32, 8), act_function ='relu'):
        input_len, output_len = env.observation_space.shape[0], env.action_space.shape[0]
        self.model = Sequential()
        self.model.add(Dense(units=layers[0], input_dim=input_len, activation=act_function))
        for i in range(0, (len(layers)-1)):
            self.model.add(Dense(layers[i+1], activation=act_function))
            self.model.add(Dropout(0.2))
        self.model.add(Dense(units=output_len, activation='linear'))
        opt = optimizers.Adam(lr=0.001, beta_1=0.8,  beta_2=0.999,
                              epsilon=1e-7, decay=0.0, amsgrad=False)
        self.model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    def train(self, data, epochs, verbose=True, val_prop=0.1, batch_size=128):
        X_train, X_val, y_train, y_val = train_test_split(data['observations'],
                                                          np.squeeze(data['actions']),
                                                          test_size=val_prop)
        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=verbose,
                       validation_data=(X_val, y_val))

    def save(self, filename):
        self.model.save(os.getcwd() + '/models/bc/' + filename + '.h5')
    # def load(self, filename):
    #     self.model.load(os.getcwd() + '/models/' + filename + '.h5')

    def save_plot(self, filename):
        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.getcwd() + '/results/bc/' + filename + '.pdf')

    def plot(self, filename):
        plt.plot(self.model.history.history['loss'])
        plt.plot(self.model.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

    def save_results(self, filename):
        with open(os.getcwd() + '/results/bc/' + filename + '.pkl', "wb") as f:
            pickle.dump(self.model.history.history, f)
