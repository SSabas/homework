# UC Berkeley Deep Reinforcement Learning Course (CS294-112, 2017)

This is my GitHub repo for homework for [CS294](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/index.html).
To note, I covered this course remotely using the lecture materials and videos offered online. 
See below the outline of the assignments.

## Dependencies

* [Python 3.5](https://www.python.org/)
* [Tensorflow 1.10.5](https://www.tensorflow.org/)
* [Numpy 1.14.5](http://www.numpy.org/)
* [OpenAI Gym 0.10.5](https://gym.openai.com/)
* [MuJoCo 1.5 and mujoco-py 1.50.1.56](http://www.mujoco.org/)

## Homework 1 - Imitation Learning

Implementation of BC (behavior cloning) and DAgger (Dataset Aggregation) methods.

## Homework 2

I implemented the policy gradient algorithm and ran some tests on various environments. I played with the hyperparameters and saw that my implementation caused the agent's reward to converge to the theoretical value. I also implemented GAE (generalized advantage estimation) and compared its results. 

## Homework 3

I implemented the DQN algorithm and ran it on the Atari Pong simulator. I experimented with different hyperparameters and saw that my model converged to the perfect value.

## Homework 4

I implemented the MPC algorithm. However, I was unable to run the provided HalfCheetahEnvNew as it threw 
