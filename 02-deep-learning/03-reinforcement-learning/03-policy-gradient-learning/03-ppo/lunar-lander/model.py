import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """Policy Network."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size

        self.policy_layer =  nn.Sequential(nn.Linear(self.state_size, 64), 
                                           nn.Tanh(), 
                                           nn.Linear(64, 64), 
                                           nn.Tanh(), 
                                           nn.Linear(64, self.action_size), 
                                           nn.Softmax(dim=-1))
        
        self.value_layer = nn.Sequential(nn.Linear(state_size, 64), 
                                         nn.Tanh(), 
                                         nn.Linear(64, 64), 
                                         nn.Tanh(), 
                                         nn.Linear(64, 1))
    
    def act(self, state):
        
        action_probs = self.policy_layer(state)
        
        return action_probs
                                    
    def evaluate(self, state):
    
        Qsa = self.value_layer(state)
        return Qsa
        