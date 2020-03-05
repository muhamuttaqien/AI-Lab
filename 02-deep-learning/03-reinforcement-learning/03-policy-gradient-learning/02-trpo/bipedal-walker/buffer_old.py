import random
import numpy as np
from collections import namedtuple

import torch

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')

class BasicBuffer(object):
    
    def __init__(self, batch_size, seed=90):
        """Initialize a BasicBuffer object."""
        
        self.experience = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self.seed = random.seed(seed)
        
        self.memory = []
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, mask):
        """Add a new experience to buffer."""
        
        self.memory.append(self.experience(state, action, reward, next_state, mask))
        
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([exp.state for exp in experiences if exp is not None])).float()
        states = states.to(device)
        
        actions = torch.from_numpy(np.vstack([exp.action for exp in experiences if exp is not None])).long()
        actions = actions.to(device)
        
        rewards = torch.from_numpy(np.vstack([exp.reward for exp in experiences if exp is not None])).float()
        rewards = rewards.to(device)
        
        next_states = torch.from_numpy(np.vstack([exp.next_state for exp in experiences if exp is not None])).float()
        next_states = next_states.to(device)
        
        masks = torch.from_numpy(np.vstack([exp.mask for exp in experiences if exp is not None]).astype(np.uint8)).float()
        masks = masks.to(device)
        
        return (states, actions, rewards, next_states, masks)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    