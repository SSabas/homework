"""
Project:
    UC Berkeley CS294 Deep Reinforcement Learning Homework I
Description:
    Run script to evaluate the performance of the behavioral cloning model
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
import tensorflow as tf
from keras.models import load_model

import load_policy
import tf_util
import models
import helpers

# -------------------- PARSE PARAMETERS AND TRAIN MODEL ------------------------- #


def main():

    # Parse variables from bash
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--to_save', type=str, default='yes',
                        help='Whether to save the results')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--verbose', type=str, default='yes')
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--layers', type=int, nargs='+', default=(100, 100, 100))
    parser.add_argument('--act_function', type=str, default='relu')
    parser.add_argument('--iter_steps', type=int, default=20)
    parser.add_argument('--to_plot', type=str, default='yes')

    args = parser.parse_args()

    env = gym.make(args.env)

    # Define the maximum number of timesteps to be evaluated in each rollout
    if args.max_timesteps is None:
        max_timesteps = env.spec.timestep_limit

    # Retrieve expert data from previous roll-outs
    with open(os.getcwd() + "/expert_data/" + args.env + ".pkl", 'rb') as f:
        expert_data = pickle.loads(f.read())
        print('Expert data loaded.')

        print('Number of samples: ', expert_data['observations'].shape[0])
        print('Dimension of observation space: ', expert_data['observations'][0].shape)
        print('Dimension of action space: ', np.squeeze(expert_data['actions'])[0].shape)

    step_size = int(expert_data['observations'].shape[0]/args.iter_steps)
    output = {}

    student = models.SupervisedModel(env,
                                     layers=args.layers,
                                     act_function=args.act_function)

    for i in range(step_size, expert_data['observations'].shape[0]+step_size, step_size):
        print('Itaration with sample size', i)

        fitting_data = {'observations':  expert_data['observations'][i-step_size:i],
                        'actions': expert_data['actions'][i-step_size:i]}
        student.train(fitting_data, args.epochs, verbose=args.verbose,
                      val_prop=args.val_prop, batch_size=args.batch_size)

        returns = []
        observations = []
        actions = []

        for j in range(args.num_rollouts):
            observation = env.reset()
            done = False
            rollout_reward = 0.0
            steps = 0

            while not done and steps < max_timesteps:
                observation = np.array(observation)

                action = student.model.predict(x=observation[None, :],
                                               verbose=False)
                observations.append(observation)
                actions.append(action)
                observation, reward, done, _ = env.step(action)
                rollout_reward += reward
                steps += 1

                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_timesteps))

            returns.append(rollout_reward)

        print('Returns', returns)
        print('Mean return', np.mean(returns))
        print('Std of return', np.std(returns))

        output[i] = {'observations': np.array(observations),
                     'actions': np.array(actions),
                     'returns': np.array(returns)}

    if args.to_save == 'yes':
        with open(os.path.join('results/bc_hyperparameter/', args.env + '.pkl'), 'wb') as f:
            pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

    if args.to_plot == 'yes':
        helpers.plot_bc_hyper_results(args.env, bc_data=output, to_save='yes')

    return output


if __name__ == '__main__':
    main()

