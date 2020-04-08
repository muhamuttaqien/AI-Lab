import math
import numpy as np

import torch
import torch.nn as nn

class ActorNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size):
        
        super(ActorNetwork, self).__init__()
        
        self.fc1_linear = nn.Linear(state_size, hidden_size)
        self.fc2_linear = nn.Linear(hidden_size, hidden_size)
        self.fc3_linear = nn.Linear(hidden_size, action_size)
        
        self.fc3_linear.weight.data.mul_(0.1)
        self.fc3_linear.bias.data.mul_(0.0)
        
    def forward(self, x):
        
        x = torch.tanh(self.fc1_linear(x))
        x = torch.tanh(self.fc2_linear(x))
        
        mu = self.fc3_linear(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        
        return mu, std
    
class CriticNetwork(nn.Module):
    
    def __init__(self, state_size):
        
        super(CriticNetwork, self).__init__()
        
        self.fc1_linear = nn.Linear(state_size, hidden_size)
        self.fc2_linear = nn.Linear(hidden_size, hidden_size)
        self.fc3_linear = nn.Linear(hidden_size, 1)
        
        self.fc3_linear.weight.data.mul_(0.1)
        self.fc3_linear.bias.data.mul_(0.0)
        
    def forward(self, x):
        
        x = torch.tanh(self.fc1_linear(x))
        x = torch.tanh(self.fc2_linear(x))
        Qsa = self.fc3_linear(x)
        
        return Qsa
    