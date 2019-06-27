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

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gym
from pytablewriter import MarkdownTableWriter

EPS = 1e-8 # epsilon constant for numeric stability

# ------------------ HELPER FUNCTIONS FOR IMITATION LEARNING -------------------- #


def plot_bc_results():

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

        writer = MarkdownTableWriter()
        writer.table_name = "add_index_column"
        writer.from_dataframe(df,add_index_column=True)

    return writer.write_table()


def plot_bc_hyper_results(env, bc_data=None, to_save='yes'):

    # Get data
    if bc_data is None:
        with open(os.getcwd() + "/results/bc_hyperparameter/" + env + ".pkl", 'rb') as f:
            bc_data = pickle.loads(f.read())
            print('BC data loaded.')

    with open(os.getcwd() + "/expert_data/" + env + ".pkl", 'rb') as f:
        expert_data = pickle.loads(f.read())
        print('Expert data loaded.')


    bc_mean = np.array([])
    bc_std = np.array([])

    expert_mean = np.mean(expert_data['returns'])
    expert_std = np.std(expert_data['returns'])

    # Retrieve results
    for i in sorted(bc_data.keys()):
        print(i)
        bc_mean = np.append(bc_mean, np.mean(bc_data[i]['returns']))
        bc_std = np.append(bc_std, np.std(bc_data[i]['returns']))

    # Draw lines
    plt.plot(sorted(bc_data.keys()), bc_mean,
             '--', color="midnightblue", label="BC Score")
    plt.plot(sorted(bc_data.keys()), [expert_mean]*len(bc_mean),
             color="darkred", label="Expert Score")

    # Draw bands
    plt.fill_between(sorted(bc_data.keys()), bc_mean - bc_std, bc_mean + bc_std,
                     color="dodgerblue", alpha=0.3)
    plt.fill_between(sorted(bc_data.keys()), [expert_mean - expert_std]*len(bc_mean), [expert_mean + expert_std]*len(bc_mean),
                     color="lightcoral", alpha=0.3)

    # Create plot
    plt.title("Learning Curve of %s" %env)
    plt.xlabel("Training Set Size"), plt.ylabel("Reward"), plt.legend(loc="best")
    plt.tight_layout()

    if to_save == 'yes':
        plt.savefig(os.getcwd() + '/results/bc_hyperparameter/' + env + '.png',
                    bbox_inches='tight')

    else:
        plt.show()

    return


def evaluate_model(env, model, max_timesteps, num_rollouts):

    # Identify the environment
    agent = gym.make(env)

    # Define the maximum number of timesteps to be evaluated in each rollout
    if max_timesteps is None:
        max_timesteps = agent.spec.timestep_limit

    # Do the roll-outs

    returns = []
    observations = []
    actions = []

    for j in range(num_rollouts):
        observation = agent.reset()
        done = False
        rollout_reward = 0.0
        steps = 0

        while not done and steps < max_timesteps:
            observation = np.array(observation)

            action = model.model.predict(x=observation[None, :], verbose=False)
            observations.append(observation)
            actions.append(action)
            observation, reward, done, _ = agent.step(action)
            rollout_reward += reward
            steps += 1

            if steps % 100 == 0:
                print("%i/%i" % (steps, max_timesteps))

        returns.append(rollout_reward)

    print('Returns', returns)
    print('Mean return', np.mean(returns))
    print('Std of return', np.std(returns))

    output = {'observations': np.array(observations),
              'actions': np.array(actions),
              'returns': np.array(returns)}

    return output


def plot_bc_dagger_comp_results(env, bc_data=None, dagger_data=None, to_save='yes'):

    # Get data
    if bc_data is None:
        with open(os.getcwd() + "/results/bc_hyperparameter/" + env + ".pkl", 'rb') as f:
            bc_data = pickle.loads(f.read())
            print('BC data loaded.')

    if dagger_data is None:
        with open(os.getcwd() + "/results/dagger_evaluation/" + env + ".pkl", 'rb') as f:
            dagger_data = pickle.loads(f.read())
            print('Dagger data loaded.')

    with open(os.getcwd() + "/expert_data/" + env + ".pkl", 'rb') as f:
        expert_data = pickle.loads(f.read())
        print('Expert data loaded.')

    bc_mean = np.array([])
    bc_std = np.array([])

    dagger_mean = np.array([])
    dagger_std = np.array([])

    expert_mean = np.mean(expert_data['returns'])
    expert_std = np.std(expert_data['returns'])

    # Retrieve results
    for i in sorted(bc_data.keys()):
        print(i)
        bc_mean = np.append(bc_mean, np.mean(bc_data[i]['returns']))
        bc_std = np.append(bc_std, np.std(bc_data[i]['returns']))

    for i in sorted(dagger_data.keys()):
        print(i)
        dagger_mean = np.append(dagger_mean, np.mean(dagger_data[i]['returns']))
        dagger_std = np.append(dagger_std, np.std(dagger_data[i]['returns']))

    # Draw lines
    plt.plot(sorted(bc_data.keys()), bc_mean,
             '--', color="midnightblue", label="BC Score")
    plt.plot(sorted(dagger_data.keys()), dagger_mean,
             color="darkolivegreen", label="Dagger Score")
    plt.plot(sorted(bc_data.keys()), [expert_mean]*len(bc_mean),
             color="darkred", label="Expert Score")

    # Draw bands
    plt.fill_between(sorted(bc_data.keys()), bc_mean - bc_std, bc_mean + bc_std,
                     color="dodgerblue", alpha=0.3)
    plt.fill_between(sorted(dagger_data.keys()), dagger_mean - dagger_std, dagger_mean + dagger_std,
                     color="darkseagreen", alpha=0.3)
    plt.fill_between(sorted(bc_data.keys()), [expert_mean - expert_std]*len(bc_mean), [expert_mean + expert_std]*len(bc_mean),
                     color="lightcoral", alpha=0.3)

    # Create plot
    plt.title("Learning Curve of %s" %env)
    plt.xlabel("Training Set Size"), plt.ylabel("Reward"), plt.legend(loc="best")
    plt.tight_layout()

    if to_save == 'yes':
        plt.savefig(os.getcwd() + '/results/dagger_evaluation/' + env + '.png',
                    bbox_inches='tight')

    else:
        plt.show()

    return
