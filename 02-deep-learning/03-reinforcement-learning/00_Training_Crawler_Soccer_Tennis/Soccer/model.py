import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, weight_scale=1.0):
    
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(weight_scale)
    nn.init.constant_(layer.bias.data, 0)
    
    return layer

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, seed):
        
        super(ActorNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1_linear = layer_init(nn.Linear(self.state_size, 256))
        self.fc2_linear = layer_init(nn.Linear(256, 128))
        self.fc3_linear = layer_init(nn.Linear(128, self.action_size))
        
    def forward(self, state, action=None):
        
        x = F.relu(self.fc1_linear(state))
        x = F.relu(self.fc2_linear(x))
        action_prob = F.softmax(self.fc3_linear(x), dim=1)
        
        dist = torch.distributions.Categorical(action_prob)
        
        if action is None:
            action = dist.sample()
            
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy().mean()

        return action, action_log_prob, dist_entropy
    
class CriticNetwork(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self, state_size, seed):
        
        super(CriticNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
            
        self.fc1_linear = layer_init(nn.Linear(self.state_size, 256))
        self.fc2_linear = layer_init(nn.Linear(256, 128))
        self.fc3_linear = layer_init(nn.Linear(128, 1))
        
    def forward(self, state):
        
        x = F.relu(self.fc1_linear(state))
        x = F.relu(self.fc2_linear(x))
        Qsa = self.fc3_linear(x)
        
        return Qsa
    