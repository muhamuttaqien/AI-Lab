import os
import gym
import argparse
import numpy as np
from collections import deque
import math
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter 
from .model import Actor, Critic,train_model
from .zfilter import ZFilter

## Set Configs

learning_rate = 1e-4
weight_decay = 1e-3
gamma = 0.95
lamda = 0.98
hidden_size = 64
clip_param = 0.2
model_update_num = 10
batch_size = 64

## Define PPO Algorithm

class Policy():

    def __init__(self,state_len,action_len,name):
        self.name=name
        self.actor = Actor(state_len, action_len, hidden_size)
        self.critic = Critic(state_len, hidden_size)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=learning_rate, 
                              weight_decay=args.l2_rate)
        self.running_state = ZFilter((state_len,), clip=5)
        self.action=0
        self.memory=deque()

    def get_action(self,states):
        state=self.running_state(states)
        mu, std = self.actor(torch.Tensor(state).unsqueeze(0))
        actions = get_act(mu, std)[0]
        return actions

    def memorize(self,states,action,reward,n_state,done):
        state=self.running_state(states)
        next_state=self.running_state(n_state)
        if done:
            mask=0
        else:
            mask=1
        self.memory.append([state, action, reward, mask])
        if done:
            self.learn()

    def learn(self):
        self.actor.train(), self.critic.train()
        train_model(self.actor, self.critic, self.memory, self.actor_optim, self.critic_optim, model_update_num, batch_size, clip_param, gamma, lamda)
        self.memory=deque()

    def save(self):
        path=self.name+".tar"
        torch.save({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
            }, filename=path)

    def load(self):
        path=self.name+".tar"
        self.actor.load_state_dict(path['actor'])
        self.critic.load_state_dict(path['critic'])

def get_act(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

