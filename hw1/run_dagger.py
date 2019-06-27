"""
Project:
    UC Berkeley CS294 Deep Reinforcement Learning Homework I
Description:
    Run script for training the DAgger model
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

import models
import helpers
import tf_util
import load_policy

# -------------------- PARSE PARAMETERS AND TRAIN MODEL ------------------------- #


def main():

    # Parse variables from bash
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument("--max_timesteps", type=int, default=None)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    parser.add_argument('--epochs', dest='epochs', type=int, default=20)
    parser.add_argument('--save_model', type=str, default='no')
    parser.add_argument('--save_results', type=str, default='yes')
    parser.add_argument('--plot', type=str, default='no')
    parser.add_argument('--verbose', type=str, default='yes')
    parser.add_argument('--val_prop', type=float, default=0.1)
    parser.add_argument('--layers', type=int, nargs='+', default=(100,100,100))
    parser.add_argument('--act_function', type=str, default='relu')
    parser.add_argument('--dagger_iter', type=int, default=20)
    parser.add_argument('--dagger_step', type=int, default=1000)

    args = parser.parse_args()

    # Retrieve expert data from previous roll-outs
    with open(os.getcwd() + "/expert_data/" + args.env + ".pkl", 'rb') as f:
        expert_data = pickle.loads(f.read())
        print('Expert data loaded.')

    env = gym.make(args.env)
    dagger_output = {}
    fitting_data = {'observations': expert_data['observations'][:args.dagger_step],
                    'actions': expert_data['actions'][:args.dagger_step]}

    # Define the maximum number of timesteps to be evaluated in each rollout
    if args.max_timesteps is None:
        max_timesteps = env.spec.timestep_limit

    # Initialise model
    dagger_model = models.SupervisedModel(env, layers=args.layers,
                                          act_function=args.act_function)
    dagger_model.train(fitting_data, args.epochs, verbose=args.verbose,
                       val_prop=args.val_prop, batch_size=args.batch_size)

    # Load expert policy
    print('Loading and building expert policy')
    expert_policy = load_policy.load_policy(os.getcwd() + '/experts/' + args.env + '.pkl')
    print('Loaded and built')

    # Evaluate the model
    dagger_output[args.dagger_step] = helpers.evaluate_model(args.env, dagger_model,
                                                             max_timesteps, args.num_rollouts)

    with tf.Session():
        tf_util.initialize()
        for i in range(1, args.dagger_iter):

            # Generate observations (i.e. run current policy for specified number of dagger steps)
            print('Dagger iterations number', i)

            observations = []
            actions = []
            expert_actions = []

            observation = env.reset()
            steps = 0

            while steps < args.dagger_step:
                observation = np.array(observation)
                action = dagger_model.model.predict(x=observation[None, :],
                                                    verbose=False)
                expert_action = expert_policy(observation[None, :])
                observations.append(observation)
                actions.append(action)
                expert_actions.append(expert_action)
                observation, reward, done, _ = env.step(action)
                steps += 1

                if steps % 100 == 0:
                    print("%i/%i" % (steps, args.dagger_step))

            # Add new observations and expert actions to the fitting data
            fitting_data['actions'] = np.vstack((fitting_data['actions'], np.array(expert_actions)))
            fitting_data['observations'] = np.vstack((fitting_data['observations'], np.array(observations)))

            # Refit the policy with new data (initialising the previous model)
            dagger_model = models.SupervisedModel(env, layers=args.layers,
                                                  act_function=args.act_function)
            dagger_model.train(fitting_data, args.epochs, verbose=args.verbose,
                               val_prop=args.val_prop, batch_size=args.batch_size)

            # Evaluate the model
            dagger_output[args.dagger_step*(i+1)] = helpers.evaluate_model(args.env, dagger_model,
                                                                     max_timesteps, args.num_rollouts)

            # Iterate it over many times
            if args.save_results == 'yes':
                with open(os.path.join('results/dagger_evaluation/', args.env + '.pkl'), 'wb') as f:
                    pickle.dump(dagger_output, f, pickle.HIGHEST_PROTOCOL)

    return dagger_output


if __name__ == '__main__':
    main()

