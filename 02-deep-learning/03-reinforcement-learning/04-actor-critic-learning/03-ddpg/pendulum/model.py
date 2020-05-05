import torch
import torch.nn as nn
import torch.nn.functional as F

    
class PolicyNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1_linear = nn.Linear(self.state_size, 512)
        self.fc2_linear = nn.Linear(512, 128)
        self.fc3_linear = nn.Linear(128, self.action_size)
        
    def forward(self, state):
        
        x = F.relu(self.fc1_linear(state))
        x = F.relu(self.fc2_linear(x))
        x = torch.tanh(self.fc3_linear(x)) # continuous action spaces
        
        return x
    
class ValueNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        
        super(ValueNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1_linear = nn.Linear(self.state_size, 1024)
        self.fc2_linear = nn.Linear(1024 + self.action_size, 512)
        self.fc3_linear = nn.Linear(512, 300)
        self.fc4_linear = nn.Linear(300, 1)
        
    def forward(self, state, action):
        
        x = F.relu(self.fc1_linear(state))
        xa_cat = torch.cat([x, action], 1)
        x_action = F.relu(self.fc2_linear(xa_cat))
        x_action = F.relu(self.fc3_linear(x_action))
        Qsa = self.fc4_linear(x_action)
        
        return Qsa
    