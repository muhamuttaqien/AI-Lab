import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class PolicyNetwork(nn.Module):
    """Policy Network."""
    
    def __init__(self, state_size, action_size, action_std, seed):
        """Initialize parameters and build model."""
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        self.action_std = action_std
        
        self.policy_layer =  nn.Sequential(nn.Linear(self.state_size, 64), 
                                           nn.Tanh(), 
                                           nn.Linear(64, 32), 
                                           nn.Tanh(), 
                                           nn.Linear(32, self.action_size), 
                                           nn.Tanh()) # continuous action spaces
        
        self.value_layer = nn.Sequential(nn.Linear(state_size, 64), 
                                         nn.Tanh(), 
                                         nn.Linear(64, 32), 
                                         nn.Tanh(), 
                                         nn.Linear(32, 1))
                
        self.action_var = torch.full((self.action_size,), self.action_std*self.action_std)
        
    def act(self, state):
        
        action_mean = self.policy_layer(state)
        action_var = self.action_var
        
        return action_mean, action_var
                                    
    def evaluate(self, state):
    
        state_value = self.value_layer(state)
        return state_value
        