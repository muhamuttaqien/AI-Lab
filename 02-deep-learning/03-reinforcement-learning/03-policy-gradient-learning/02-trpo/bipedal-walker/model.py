import torch
import torch.autograd as autograd
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Policy Network."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.affine1_layer = nn.Linear(self.state_size, 128)
        self.affine2_layer = nn.Linear(128, 64)
        
        self.action_mean = nn.Linear(64, self.action_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        
        self.action_log_std = nn.Parameter(torch.zeros(1, self.action_size))
        
        self.saved_actions = []
        self.rewards = []
        self.final_value = 0
        
    def forward(self, state):
        
        x = torch.tanh(self.affine1_layer(state))
        x = torch.tanh(self.affine2_layer(x))
        
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_log_std, action_std
    
class ValueNetwork(nn.Module):
    """Value Network."""
    
    def __init__(self, state_size, seed):
        """Initialize parameters and build model."""
        
        super(ValueNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        
        self.affine1_layer = nn.Linear(self.state_size, 128)
        self.affine2_layer = nn.Linear(128, 64)
        
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        
    def forward(self, x):
        
        x = torch.tanh(self.affine1_layer(x))
        x = torch.tanh(self.affine2_layer(x))
        
        Qsa = self.value_head(x)
        
        return Qsa
    