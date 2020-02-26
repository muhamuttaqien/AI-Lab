import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """Define DQN architecture."""
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model."""
        
        super(DQN, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1_layer = nn.Linear(state_size, fc1_units)
        self.fc2_layer = nn.Linear(fc1_units, fc2_units)
        self.fc3_layer = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        """Build a network that maps state into action values."""
        
        x = F.relu(self.fc1_layer(state))
        x = F.relu(self.fc2_layer(x))
        Qsa = self.fc3_layer(x)
        
        return Qsa
    