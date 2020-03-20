import random
import numpy as np
from collections import namedtuple

import torch

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')
    
class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, batch_size, seed):
        """Initialize a ReplayMemory object."""
        
        self.experience = namedtuple('Experience', field_names=['actor_state', 'critic_state', 'action', 'log_prob', 'reward'])        
        self.seed = np.random.seed(seed)
        
        self.memory = []

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to buffer."""
        
        self.memory.append(self.experience(actor_state, critic_state, action, log_prob, reward))
        
    def get_experiences(self, clear=True):
        """Get collected experiences."""
        
        actor_states = np.vstack([exp.actor_state for exp in self.memory if exp is not None])
        critic_states = np.vstack([exp.critic_state for exp in self.memory if exp is not None])
        actions = np.vstack([exp.action for exp in self.memory if exp is not None])
        log_probs = np.vstack([exp.log_prob for exp in self.memory if exp is not None])
        rewards = np.vstack([exp.reward for exp in self.memory if exp is not None])
        
        num_experiences = len(self)
        
        if clear: self.clear()
            
        return (actor_states, critic_states, actions, log_probs, rewards, num_experiences)
    
    def delete(self, index):
        """Delete experience by index."""
        del self.memory[index]
        
    def clear(self):
        """Clear all collected experiences."""
        self.memory.clear()
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    