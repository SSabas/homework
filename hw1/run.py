"""
Project:
    UC Berkeley CS294 Deep Reinforcement Learning Homework I
Description:
    Run script for behaviour cloning and imitation learning
Authors:
  Sven Sabas
Date:
  25/06/2019
"""

# ----------------------------- IMPORT LIBRARIES -------------------------------- #

import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import importlib
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.optimizers import SGD

import helpers
import load_policy
import tf_util

# ------------------------ BUILD KERAS NN MODEL --------------------------------- #









agent = 'HalfCheetah-v2'
env = gym.make(agent)
filename = "/Users/ssabas/Desktop/ucbcs294/hw1/expert_data/" + agent + ".pkl"

with open(filename, "rb") as f:
    data = pickle.load(f)

student = SupervisedModel(env)
student.train(data, 30)

student.plot(agent)


student.save(agent)


student.plot(agent)
student2

expert_policy_file = "/Users/ssabas/Desktop/ucbcs294/hw1/experts/" + agent + ".pkl"
expert_policy = load_policy.load_policy(expert_policy_file)


history = student.model.history


"""
Project:
    UC Berkeley CS294 Deep Reinforcement Learning Homework I
Description:
    Run script for behavioral cloning
Authors:
  Sven Sabas
Date:
  25/06/2019
"""


# ----------------------------- IMPORT LIBRARIES -------------------------------- #
import pickle
import os
import tensorflow as tf
import numpy as np
import gym
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib
from sklearn.utils import shuffle
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Reshape


import tf_util
import models


def main():

    # Parse variables from bash
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--epochs', dest='epochs', type=int, default=100)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=5,
                        help='Number of expert roll outs')
    parser.add_argument('--fit_model', type=str, default='no')
    parser.add_argument('--save_model', type=str, default='no')
    parser.add_argument('--plot', type=str, default='yes')
    parser.add_argument('--verbose', type=str, default='yes')
    parser.add_argument('--val_prop', type=float, default=0.1)

    args = parser.parse_args()

    # Retrieve expert data from previous roll-outs
    with open(os.getcwd() + "/expert_data/" + args.env + ".pkl", 'rb') as f:
        expert_data = pickle.loads(f.read())
        print('Expert data loaded.')

    if args.fit_model == 'yes':
        print('Number of samples: ', expert_data['observations'].shape[0])
        print('Dimension of observation space: ', expert_data['observations'][0].shape)
        print('Dimension of action space: ', np.squeeze(expert_data['actions'])[0].shape)

        agent = gym.make(args.env)
        student = models.SupervisedModel(agent)
        student.train(expert_data, args.epochs, verbose=args.verbose,
                      val_prop=args.val_prop, batch_size=args.batch_size)

        if args.save_model == 'yes':
            student.save(args.env)
            print('Model saved.')


        if args.plot == 'yes':
            student.plot(args.env)
            print('Model performance plot saved')

    else:
        student = load_model(os.getcwd() + '/models/' + args.envname + '.h5')
        print('Model loaded.')


    # # set up session
    # tfconfig = tf.ConfigProto()
    # tfconfig.gpu_options.allow_growth = True
    # num_train = obs_train.shape[0]
    # shuffle_list = np.arange(num_train)
    # losses = []
    # with tf.Session(config=tfconfig) as sess:
    #     # model = Behavioral_clone(obs.shape[1], acts.shape[1])
    #     # model.build_net([128, 256, 512, 256, 128], lr=args.lr)
    #     # sess.run(tf.global_variables_initializer())
    #
    #     tf_util.initialize()
    #
    #     env = gym.make(args.envname)
    #     max_steps = args.max_timesteps or env.spec.timestep_limit
    #
    #     returns = []
    #     observations = []
    #     actions = []
    #     model = load_model('output/' + args.envname + '_bc.h5')
    #     for i in range(args.num_rollouts):
    #         print('iter', i)
    #         obs = env.reset()
    #         done = False
    #         totalr = 0.
    #         steps = 0
    #         while not done:
    #             # action = model.action(sess, obs[None, :])
    #             obs = obs.reshape(1, len(obs))
    #             action = (model.predict(obs, batch_size=64, verbose=0))
    #             observations.append(obs)
    #             actions.append(action)
    #             obs, r, done, _ = env.step(action)
    #             totalr += r
    #             steps += 1
    #             if args.render:
    #                 env.render()
    #             if steps % 100 == 0:
    #                 print("%i/%i" % (steps, max_steps))
    #             if steps >= max_steps:
    #                 break
    #         returns.append(totalr)
    #
    #     print('returns', returns)
    #     print('mean return', np.mean(returns))
    #     print('std of return', np.std(returns))


if __name__ == '__main__':
    main()

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
    student = load_model(os.getcwd() + '/models/bc/' + 'Humanoid-v2' + '.h5')



    # Load expert policy
    expert_policy = load_policy.load_policy(os.getcwd() + '/experts/' + args.env + '.pkl')
    expert_policy = load_policy.load_policy(os.getcwd() + '/experts/' + 'Humanoid-v2' + '.pkl')



    returns = []
    observations = []
    actions = []
    expert_actions = []

    with tf.Session():
        tf_util.initialize()
        for i in range(num_rollouts):
            observation = env.reset()
            done = False
            rollout_reward = 0.0
            steps = 0

            while not done and steps < max_timesteps:
                observation = np.array(observation)

                action = student.model.predict(x=observation[None, :],
                                               verbose=False)
                expert_action = expert_policy(observation[None, :])

                observations.append(observation)
                actions.append(action)
                expert_actions.append(expert_action) # expert labeling
                observation, reward, done, _ = env.step(action)
                rollout_reward += reward
                steps += 1

                if render:
                    env.render()
                if steps % 100 == 0:
                    print("%i/%i" % (steps, max_timesteps))

            returns.append(rollout_reward)

    print('Returns', returns)
    print('Mean return', np.mean(returns))
    print('Std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                   'returns': np.array(returns),
                   'expert_actions': np.array(expert_actions)}

    if args.to_save == 'yes':
        with open(os.path.join('results/bc_evaluation/', env + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

    observations = np.array(observations, dtype=np.float32)
    expert_actions = np.array(expert_actions, dtype=np.float32).squeeze()
    actions = np.array(actions, dtype=np.float32).squeeze()

    return observations, actions, returns, expert_actions


if __name__ == '__main__':
    main()

