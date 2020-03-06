import random
import numpy as np
from collections import namedtuple

import torch

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')

    
class BasicBuffer(object):
    
    def __init__(self, seed=90):
        """Initialize a BasicBuffer object."""
        
        self.experience = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'mask'))
        self.seed = random.seed(seed)
        
        self.memory = []

    def add(self, state, action, reward, next_state, mask):
        """Add a new experience to buffer."""
            
        self.memory.append(self.experience(state, action, reward, next_state, mask))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        return self.experience(*zip(*self.memory))

    def clear_memory(self):
        """Clear all experiences from memory."""
        del self.memory[:]
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
