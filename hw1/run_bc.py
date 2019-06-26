"""
Project:
    UC Berkeley CS294 Deep Reinforcement Learning Homework I
Description:
    Run script for training the behavioral cloning model
Authors:
  Sven Sabas
Date:
  25/06/2019
"""


# ----------------------------- IMPORT LIBRARIES -------------------------------- #

import pickle
import os
import numpy as np
import gym
import argparse

import models

# -------------------- PARSE PARAMETERS AND TRAIN MODEL ------------------------- #


def main():

    # Parse variables from bash
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--save_model', type=str, default='no')
    parser.add_argument('--save_results', type=str, default='no')
    parser.add_argument('--plot', type=str, default='no')
    parser.add_argument('--verbose', type=str, default='yes')
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--layers', type=int, nargs='+', default=(16,32,8))
    parser.add_argument('--act_function', type=str, default='relu')

    args = parser.parse_args()

    # Retrieve expert data from previous roll-outs
    with open(os.getcwd() + "/expert_data/" + args.env + ".pkl", 'rb') as f:
        expert_data = pickle.loads(f.read())
        print('Expert data loaded.')

        print('Number of samples: ', expert_data['observations'].shape[0])
        print('Dimension of observation space: ', expert_data['observations'][0].shape)
        print('Dimension of action space: ', np.squeeze(expert_data['actions'])[0].shape)

        agent = gym.make(args.env)
        student = models.SupervisedModel(agent,
                                         layers=args.layers,
                                         act_function=args.act_function)
        student.train(expert_data, args.epochs, verbose=args.verbose,
                      val_prop=args.val_prop, batch_size=args.batch_size)

        if args.save_model == 'yes':
            student.save(args.env)
            print('Model saved.')

        if args.plot == 'yes':
            student.save_plot(args.env)
            print('Model performance plot saved')

        if args.save_results == 'yes':
            student.save_results(args.env)
            print('Model fitting history saved')


if __name__ == '__main__':
    main()

