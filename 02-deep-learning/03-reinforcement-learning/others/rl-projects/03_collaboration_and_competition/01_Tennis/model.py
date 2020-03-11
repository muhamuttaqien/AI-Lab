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
        self.fc2_linear = nn.Linear(256, 256)
        self.fc3_linear = nn.Linear(256, self.action_size)

        self.normalizer = nn.BatchNorm1d(256)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1_linear.weight.data.uniform_(*hidden_init(self.fc1_linear))
        self.fc2_linear.weight.data.uniform_(*hidden_init(self.fc2_linear))
        self.fc3_linear.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        
        if states.dim() == 1:
            states = torch.unsqueeze(states, 0)
        
        x = F.relu(self.fc1_linear(states))
        x = self.normalizer(x)
        x = F.relu(self.fc2_linear(x))
        x = F.tanh(self.fc3_linear(x))
        
        return x

class ValueNetwork(nn.Module):
    """Value (Critic) Network."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        
        super(ValueNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
                
        self.fc1_linear = nn.Linear(self.state_size, 256)
        self.fc2_linear = nn.Linear(256, 256)
        self.fc3_linear = nn.Linear(256, 1)
        
        self.normalizer = nn.BatchNorm1d(256)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.fc1_linear.weight.data.uniform_(*hidden_init(self.fc1_linear))
        self.fc2_linear.weight.data.uniform_(*hidden_init(self.fc2_linear))
        self.fc3_linear.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1_layer(xs))
        x = self.normalizer(x)
        x = F.relu(self.fc2_linear(x))
        Qsa = self.fc3_linear(x)
        
        return Qsa

class ActorCriticNetwork():
    """Object containing all models required per DDPG agent."""
    
    def __init__(self, num_agents, state_size, action_size, seed):
        
        self.actor = PolicyNetwork(state_size, action_size, seed)
        self.actor_target = PolicyNetwork(state_size, action_size, seed)
        
        critic_input_size = (state_size+action_size) * num_agents
        self.critic = ValueNetwork(critic_input_size, action_size, seed)
        self.critic_target = ValueNetwork(critic_input_size, action_size, seed)
        