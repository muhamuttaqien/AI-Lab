import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    
    fan_init = layer.weight.data.size()[0]
    limit = 1. / np.sqrt(fan_init)
    return (-limit, limit)

class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, seed):
        
        super(ActorNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.bn1 = nn.BatchNorm1d(self.state_size)
        self.fc1_linear = nn.Linear(self.state_size, 128)
        
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2_linear = nn.Linear(128, 128)
        
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3_linear = nn.Linear(128, 128)

        self.bn4 = nn.BatchNorm1d(128)
        self.fc4_linear = nn.Linear(128, self.action_size)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        
        self.fc1_linear.weight.data.uniform_(*hidden_init(self.fc1_linear))
        self.fc2_linear.weight.data.uniform_(*hidden_init(self.fc2_linear))
        self.fc3_linear.weight.data.uniform_(*hidden_init(self.fc3_linear))
        self.fc4_linear.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        
        x = F.relu(self.fc1_linear(self.bn1(state)))
        x = F.relu(self.fc2_linear(self.bn2(x)))
        x = F.relu(self.fc3_linear(self.bn3(x)))
        action_means = F.tanh(self.fc4_linear(self.bn4(x)))
        
        return action_means
    
class CriticNetwork(nn.Module):
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size, seed):
        
        super(CriticNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.bn1 = nn.BatchNorm1d(self.state_size)
        self.fc1_linear = nn.Linear(self.state_size, 128)
        
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2_linear = nn.Linear(128, 128)
        
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3_linear = nn.Linear(128, 128)

        self.bn4 = nn.BatchNorm1d(128)
        self.fc4_linear = nn.Linear(128, 1)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        
        self.fc1_linear.weight.data.uniform_(*hidden_init(self.fc1_linear))
        self.fc2_linear.weight.data.uniform_(*hidden_init(self.fc2_linear))
        self.fc3_linear.weight.data.uniform_(*hidden_init(self.fc3_linear))
        self.fc4_linear.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        
        x = F.relu(self.fc1_linear(self.bn1(state)))
        x = F.relu(self.fc2_linear(self.bn2(x)))
        x = F.relu(self.fc3_linear(self.bn3(x)))
        Qsa = F.tanh(self.fc4_linear(self.bn4(x)))
        
        return Qsa
    
class PolicyNetwork(nn.Module):
    """Policy Network."""
    
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model."""
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
            
        self.actor = ActorNetwork(self.state_size, self.action_size, seed)
        self.critic = CriticNetwork(self.state_size, self.action_size, seed)
        self.action_std = nn.Parameter(torch.ones(1, self.action_size)*0.15)

    def forward(self, states, actions=None, scale=1.):

        action_means = self.actor(states)
        state_values = self.critic(states)
        
        dist = torch.distributions.Normal(action_means, F.hardtanh(self.action_std, min_val=0.06*scale, max_val=0.6*scale))

        if actions is None:
            actions = dist.sample()
            
        action_log_probs = dist.log_prob(actions)
        action_log_probs = torch.sum(action_log_probs, dim=1, keepdim=True)

        dist_entropies = dist.entropy().mean()

        return actions, action_log_probs, state_values, dist_entropies
