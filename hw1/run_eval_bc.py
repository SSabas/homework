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

# -------------------- PARSE PARAMETERS AND TRAIN MODEL ------------------------- #


def main():

    # Parse variables from bash
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--to_save', type=str, default='yes',
                        help='Whether to save the results')

    args = parser.parse_args()

    # Get the expert policy model
    env = gym.make(args.env)

    # Define the maximum number of timesteps to be evaluated in each rollout
    if args.max_timesteps is None:
        max_timesteps = env.spec.timestep_limit

    # Get the cloned model
    student = load_model(os.getcwd() + '/models/bc/' + args.env + '.h5')

    returns = []
    observations = []
    actions = []

    for i in range(args.num_rollouts):
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

            if args.render:
                env.render()
            if steps % 100 == 0:
                print("%i/%i" % (steps, max_timesteps))

        returns.append(rollout_reward)

    print('Returns', returns)
    print('Mean return', np.mean(returns))
    print('Std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                   'returns': np.array(returns)}

    if args.to_save == 'yes':
        with open(os.path.join('results/bc_evaluation/', args.env + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32).squeeze()

    return observations, actions, returns


if __name__ == '__main__':
    main()

