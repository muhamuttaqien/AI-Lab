import os
import math
import numpy as np

from collections import deque

import torch
import torch.optim as optim

from .zfilter import ZFilter

from .utils import log_prob_density, surrogate_loss

from .model import ActorNetwork, CriticNetwork

## Set Configs

learning_rate = 1e-4
weight_decay = 1e-3
gamma = 0.95
lamda = 0.98
hidden_size = 64
clip = 0.2
update_every = 10
batch_size = 64

## Define PPO Algorithm

class Agent():
    
    def __init__(self, state_size, action_size, hidden_size, name):
        
        self.name = name
        
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.lamda = lamda
        self.hidden_size = hidden_size
        self.clip = clip
        self.update_every = update_every
        self.batch_size = batch_size
        
        self.actor = ActorNetwork(state_size, action_size, hidden_size)
        self.critic = CriticNetwork(state_size, hidden_sizeˇˇ)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        self.running_state = ZFilter((state_size,), clip=self.clip)
        self.memory = deque()        
        
        self.action = 0
        
    def memorize(self, states, actions, rewards, next_states, dones):
        
        states = self.running_state(states)
        next_states = self.running_state(next_states)
        
        if done: mask = 0
        else: mask = 1
            
        self.memory.append([state, action, reward, mask])
        
        if done: self.learn()
        
    def act(self, states):
        
        states = self.running_state(states)
        mu, std = self.actor(torch.Tensor(states).unsqueeze(0))
        actions = torch.normal(mu, std)
        actions = actions.data.numpy()
        
        return actions
    
    def learn(self):
        
        memory = np.array(self.memory)
        states = np.vstack(memory[:, 0])
        actions = list(memory[:, 1])
        rewards = list(memory[:, 2])
        masks = list(memory[:, 3])
        
        self.actor.train()
        
        mu, std = self.actor(torch.Tensor(states))
        old_policy = log_prob_density(torch.Tensor(actions), mu, std)
        
        self.critic.train()
        
        old_values = self.critic(torch.Tensor(states))
        returns, advantages = get_gae(rewards, masks, old_values)
        
        criterion = torch.nn.MSELoss()
        
        num_states = len(states)
        all_states = np.arange(num_states)
        
        for _ in range(self.update_every):
            
            np.random.shuffle(all_states)
            
            for i in range(num_states // self.batch_size):
                
                batch_index = all_states[self.batch_size * i : self.batch_size * (i + 1)]
                batch_index = torch.LongTensor(batch_index)
                
                inputs = torch.Tensor(states)[batch_index]
                action_samples = torch.Tensor(actions)[batch_index]
                return_samples = returns.unsqueeze(1)[batch_index]
                advantage_samples = advantages.unsqueeze(1)[batch_index]
                old_value_samples = old_values[batch_index].detach()
                
                values = self.critic(inputs)
                clipped_values = old_value_samples + torch.clamp(values - old_value_samples, 
                                                                 -self.clip, self.clip)
                
                critic_loss1 = criterion(clipped_values, return_samples)
                critic_loss2 = criterion(values, return_samples)
                
                critic_loss = torch.max(critic_loss1, critic_loss2).mean()
                
                loss, ratio = surrogate_loss(self.actor, advantage_samples, inputs, 
                                             old_policy.detach(), action_samples, batch_index)
                
                clipped_ratio = torch.clamp(ratio, 1.0 - self.clop, 1.0 + self.clip)
                
                clipped_loss = clipped_ratio * advantage_samples
                actor_loss = -torch.min(loss, clipped_loss).mean()
                
                loss = actor_loss + 0.5 * critic_loss
                
                actor_optimizer.zero_grad()
                loss.backward()
                actor_optimizer.step()
                
                critic_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()
        
    def get_gae(rewards, masks, values):
        
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        running_returns = 0
        previous_value = 0
        running_advantages = 0
        
        for i in reversed(range(0, len(rewards))):
            
            running_returns = rewards[i] + (self.gamma * running_returns * masks[i])
            
            returns[i] = running_returns
            
            running_delta = rewards[i] + (self.gamma * previous_value * masks[i]) - values.data[i]
            
            previous_value = values.data[i]
            
            running_advantages = running_delta + (self.gamma * self.lamda * running_advantages * masks[i])
            
            advantages[i] = running_advantages

        advantages = (advantages - advantages.mean()) / advantages.std()
        
        return returns, advantages
    
    def save(self, path):
        
        torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict()}, filename=path)
        
    def load(self, path):
        
        self.actor.load_state_dict(path['actor'])
        self.critic.load_state_dict(path['critic'])
        