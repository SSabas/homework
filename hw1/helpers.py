"""
Project:
    UC Berkeley CS294 Deep Reinforcement Learning Homework I
Description:
    Helper functions for imitation learning
Authors:
  Sven Sabas
Date:
  25/06/2019
"""

# ----------------------------- IMPORT LIBRARIES -------------------------------- #

import json
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from pytablewriter import MarkdownTableWriter

EPS = 1e-8 # epsilon constant for numeric stability

# ------------------ HELPER FUNCTIONS FOR IMITATION LEARNING -------------------- #


def load_expert_data(filename, verbose=False):
    """ Load the expert data from pickle saved in filename"""

    expert_data = None
    with open(filename, "rb") as f:
        expert_data = pickle.load(f)

    observations = expert_data["observations"].astype('float32')
    actions = np.squeeze(expert_data["actions"].astype('float32'))
    returns = expert_data["returns"].astype('float32')

    if verbose:
        # As a sanity check, print out the size of the training and test data.
        print('observations shape: ', observations.shape)
        print('actions shape: ', actions.shape)
        print('mean return: ', np.mean(returns))

    return observations, actions, returns


def train_test_val_split(X, y,
                         train_prop, val_prop,
                         verbose=True):
    """ Split the dataset (X, y) into train, validation, and test sets.
    NB! Proportions should sum to 1.
    Arguments:
    X -- Feature matrix, shape of (N, D_in)
    y -- Labels, shape of (N, num_classes)
    train_prop -- Proportion of training data
    val_prop -- Proportion of validation data
    """
    N_total = X.shape[0]
    N_train = int(np.floor(N_total * train_prop))
    N_val   = int(np.ceil(N_total * val_prop))
    N_test  = N_total - N_train - N_val

    assert(N_train + N_val + N_test == N_total)

    # Split the data into test set and temporary set, which will be
    # split into training and validation sets
    X_tmp, X_test, y_tmp, y_test = train_test_split(X,
                                                    y,
                                                    test_size=N_test)

    # Split X_tmp into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_tmp,
                                                      y_tmp,
                                                      train_size=N_train)

    if verbose:
        print('Train data: ', X_train.shape, X_train.dtype)
        assert(X_train.shape[0] == N_train)
        print('Train labels: ', y_train.shape, y_train.dtype)
        assert(y_train.shape[0] == N_train)

        print('Validation data: ', X_val.shape, X_val.dtype)
        assert(X_val.shape[0] == N_val)
        print('Validation labels: ', y_val.shape, y_val.dtype)
        assert(y_val.shape[0] == N_val)

        print('Test data: ', X_test.shape, X_test.dtype)
        assert(X_test.shape[0] == N_test)
        print('Test labels: ', y_test.shape, y_test.dtype)
        assert(y_test.shape[0] == N_test)

    data = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "X_dev": X_dev, "y_dev": y_dev
    }

    return data


def plot_bc_results(markdown_format='yes'):

    # Get environment names
    envs = ['Ant-v2', 'HalfCheetah-v2',
            'Hopper-v2', 'Humanoid-v2',
            'Reacher-v2', 'Walker2d-v2']

    # Create pandas dataframe
    df = pd.DataFrame(index=envs, columns=['Mean(BC)', 'Std(BC)', 'Mean(Exp)', 'Std(Exp)'])

    for i in range(len(envs)):
        print(i)

        # Get expert data data
        with open(os.getcwd() + "/expert_data/" + envs[i] + ".pkl", 'rb') as f:
            expert_data = pickle.loads(f.read())
            print('Expert data loaded.')

        # Get bc data
        with open(os.getcwd() + "/results/bc_evaluation/" + envs[i] + ".pkl", 'rb') as f:
            bc_data = pickle.loads(f.read())
            print('BC data loaded.')

        df.loc[envs[i]] = [np.mean(bc_data['returns']), np.std(bc_data['returns']), np.mean(expert_data['returns']), np.std(expert_data['returns'])]

    if markdown_format == 'yes':
        writer = MarkdownTableWriter()
        writer.table_name = "add_index_column"
        writer.from_dataframe(df,add_index_column=True)

        return writer.write_table()

    return print(df)