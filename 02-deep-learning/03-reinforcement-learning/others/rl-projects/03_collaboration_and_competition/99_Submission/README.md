# Deep Reinforcement Learning : Collaboration and Competition

This project repository is my work on 3rd Project: Collaboration and Competition for Udacity's [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).


## Project's Description

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

## The Environment

The environment is provided by [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents). For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment on the Unity ML-agents Github page.

After checking how the environment provided by Unity behaves, here are the details about the rewards, state & action spaces:

#### Rewards

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically, After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode. The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

#### State Space

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.

#### Action Space

Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

## Project Installation

#### Step 1: Clone the DRLND Repository
1. Configure your Python environment by following instructions described in the [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). The instructions can be found in the [Readme.md](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Readme.md) file.
2. By following the instructions, then you will have your own PyTorch, the ML-agents toolkit, and all the Python packages required to complete the project.
3. (For Windows users) The ML-agents toolkit supports Windows 10.

#### Step 2: Download the Unity Environment
For running this project you need to install the Unity environment as completely described in the [Getting Started section](https://github.com/udacity/deep-reinforcement-learning/blob/master/p2_continuous-control/README.md).

#### Step 3: Explore the Environment
After you have followed two installment steps above, then you can explore the project environment by opening `Collaboration_and_Competition_MADDPG.ipynb` located on the root repository. You can follow the instructions to learn how to use the Python API to control the agent.

## Train The Agent
You can try to train the agent yourself by executing the provided jupyter notebook within this project repository. By working in your local environment, the workspace will allow you to see the simulator of the environment and watch directly how our trained agent smartly behaves with the environment.

## Discussions
📨 if any discussion, please contact me anytime: muha.muttaqien@gmail.com