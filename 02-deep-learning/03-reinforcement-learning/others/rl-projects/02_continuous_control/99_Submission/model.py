import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    
    fan_init = layer.weight.data.size()[0]
    limit = 1. / np.sqrt(fan_init)
    return (-limit, limit)

class PolicyNetwork(nn.Module):
    """Policy (Actor) Network."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.normalizer = nn.BatchNorm1d(self.state_size)
        
        self.fc1_linear = nn.Linear(self.state_size, 512)
        self.fc2_linear = nn.Linear(512, 256)
        self.fc3_linear = nn.Linear(256, self.action_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1_linear.weight.data.uniform_(*hidden_init(self.fc1_linear))
        self.fc2_linear.weight.data.uniform_(*hidden_init(self.fc2_linear))
        self.fc3_linear.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, states):
        
        x = self.normalizer(states)

        x = F.relu(self.fc1_linear(x))
        x = F.relu(self.fc2_linear(x))
        x = torch.tanh(self.fc3_linear(x))
        
        return x
    
    
class ValueNetwork(nn.Module):
    """Value (Critic) Network."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        
        super(ValueNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.dropout = nn.Dropout(p=0.2)
        self.normalizer = nn.BatchNorm1d(self.state_size)
        
        self.fcs1_linear = nn.Linear(self.state_size, 512)
        self.fc2_linear = nn.Linear(512+self.action_size, 256)
        self.fc3_linear = nn.Linear(256, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fcs1_linear.weight.data.uniform_(*hidden_init(self.fcs1_linear))
        self.fc2_linear.weight.data.uniform_(*hidden_init(self.fc2_linear))
        self.fc3_linear.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, states, actions):
        
        x = self.normalizer(states)
        
        xs = F.relu(self.fcs1_linear(x))
        x = torch.cat((xs, actions), dim=1)
        x = F.relu(self.fc2_linear(x))
        x = self.dropout(x)
        Qsa = self.fc3_linear(x)
        
        return Qsa
    