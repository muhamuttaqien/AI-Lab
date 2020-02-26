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
        
        self.fc1_linear = nn.Linear(self.state_size, 256)
        self.fc2_linear = nn.Linear(256, self.action_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1_linear.weight.data.uniform_(*hidden_init(self.fc1_linear))
        self.fc2_linear.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        
        x = F.relu(self.fc1_linear(state))
        x = torch.tanh(self.fc2_linear(x))
        
        return x
    
    
class ValueNetwork(nn.Module):
    """Value (Critic) Network."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        
        super(ValueNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fcs1_linear = nn.Linear(state_size, 256)
        self.fc2_linear = nn.Linear(256+action_size, 256)
        self.fc3_linear = nn.Linear(256, 128)
        self.fc4_linear = nn.Linear(128, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fcs1_linear.weight.data.uniform_(*hidden_init(self.fcs1_linear))
        self.fc2_linear.weight.data.uniform_(*hidden_init(self.fc2_linear))
        self.fc3_linear.weight.data.uniform_(*hidden_init(self.fc3_linear))
        self.fc4_linear.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state, action):
        
        xs = F.leaky_relu(self.fcs1_linear(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2_linear(x))
        x = F.leaky_relu(self.fc3_linear(x))
        Qsa = self.fc4_linear(x)
        
        return Qsa