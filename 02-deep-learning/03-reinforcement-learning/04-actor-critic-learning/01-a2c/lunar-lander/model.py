import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, seed):
        
        super(PolicyNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1_layer = nn.Linear(input_size, hidden_size)
        self.fc2_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        
        logits = F.relu(self.fc1_layer(state))
        logits = self.fc2_layer(logits)
        probs = F.softmax(logits, dim=0)
        
        return probs
    
class ValueNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, seed):
        
        super(ValueNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.fc1_layer = nn.Linear(input_size, hidden_size)
        self.fc2_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        
        value = F.relu(self.fc1_layer(state))
        value = self.fc2_layer(value)
        
        return value
    
class HybridNetwork(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, seed):
        
        super(HybridNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.policy1_layer = nn.Linear(input_size, hidden_size)
        self.policy2_layer = nn.Linear(hidden_size, output_size)
        
        self.value1_layer = nn.Linear(input_size, hidden_size)
        self.value2_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        
        logits = F.relu(self.policy1_layer(state))
        logits = self.policy2_layer(logits)
        probs = F.softmax(logits, dim=0)
        
        value = F.relu(self.value1_layer(state))
        value = self.value2_layer(value)
        
        return probs, value
