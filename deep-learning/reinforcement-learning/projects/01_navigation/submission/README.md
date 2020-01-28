# Deep Reinforcement Learning : Navigation

This project repository is my work on 1st Project: Navigation for Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project's Description
In this navigation project, I have to train an agent to navigate on a large square world and collect yellow bananas. The goal is to collect as many yellow bananas as possible while at the same time avoiding blue bananas. The task is episodic, and in order to solve the environment, the agent must get an average score of at least +13 over 100 consecutive episodes.

![In Project 1, train an agent to navigate a large world.](./images/yellow_bananas_collector.gif)

## The Environment

The environment is provided by [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). The project environment used for this project is similar to one of the example environments, but not identical to the [Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment on the Unity ML-agents Github page.

After checking how the environment provided by Unity behaves, here are the details about the rewards, state & action spaces:

#### Rewards

1. The agent is given a reward of +1 for successfully collecting a yellow banana]
2. Reward of -1 for collecting a blue banana

#### State Space

The environment contains 37 dimensions and the agent's velocity, along with ray-based perception of objects around the agent's forward direction.

#### Action Space

There are four possible discreate actions that agent can perform including 0 for move forward, 1 for move backward, 2 for turn left and 3 for turn right.

## Project Installment

#### Step 1: Clone the DRLND Repository
1. Configure your Python environment by following instructions described in the [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). The instructions can be found in the [Readme.md](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md) file.
2. By following the instructions, then you will have your own PyTorch, the ML-agents toolkit, and all the Python packages required to complete the project.
3. (For Windows users) The ML-agents toolkit supports Windows 10.
