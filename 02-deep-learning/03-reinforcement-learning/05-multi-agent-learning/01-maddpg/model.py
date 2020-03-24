import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd



class ActorNetwork(nn.Module):
    
    def __init__(self, state_size, action_size):
        
        super(ActorNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1_linear = nn.Linear(self.state_size, 512)
        self.fc2_linear = nn.Linear(512, 128)
        self.fc3_linear = nn.Linear(128, self.action_size)
        
    def forward(self, states):
        
        x = F.relu(self.fc1_linear(states))
        x = F.relu(self.fc2_linear(x))
        x = torch.tanh(self.fc3_linear(x))
        
        return x

class CriticNetwork(nn.Module):
    
    def __init__(self, state_size, action_size):
        
        super(CriticNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1_linear = nn.Linear(self.state_size, 1024)
        self.fc2_linear = nn.Linear(1024 + self.action_size, 512)
        self.fc3_linear = nn.Linear(512, 300)
        self.fc4_linear = nn.Linear(300, 1)
        
    def forward(self, states, actions):
        
        x = F.relu(self.fc1_linear(states))
        xs = torch.cat([x, actions], dim=1)
        x = F.relu(self.fc2_linear(xs))
        x = F.relu(self.fc3_linear(x))
        Qsa = self.fc4_linear(xa)
       
        return Qsa

class ActorCriticNetwork():
    """Object containing all models required per DDPG agent."""
    
    def __init__(self, num_agents, state_size, action_size, seed):
        
        self.actor = PolicyNetwork(state_size, action_size, seed)
        self.actor_target = PolicyNetwork(state_size, action_size, seed)
        
        critic_input_size = state_size * num_agents
        self.critic = ValueNetwork(critic_input_size, action_size, seed)
        self.critic_target = ValueNetwork(critic_input_size, action_size, seed)
