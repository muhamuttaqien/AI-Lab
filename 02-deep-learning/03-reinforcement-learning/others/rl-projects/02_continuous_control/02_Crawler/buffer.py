import random
import numpy as np

import torch

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')

class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, batch_size, num_agents, seed):
        """Initialize a ReplayMemory object."""
        
        self.seed = np.random.seed(seed)
        
        self.memory = []
        self.batch_size = batch_size
        
    def add(self, trajectory):
        """Add a new trajectory to memory."""
        
        self.memory.extend(trajectory)
        
    def sample(self):
        """Randomly sample a batch of trajectories from memory."""
        
        states, actions, log_probs_old, returns, advantages = map(lambda x: torch.cat(x, dim=0), zip(*self.memory))
        advantages = (advantages - advantages.mean()) / advantages.std()
        
        indices = np.arange(states.size()[0])
        np.random.shuffle(indices)
        indices = [indices[div*self.batch_size: (div+1)*self.batch_size] for div in range(len(indices) // self.batch_size + 1)]
        
        result = []
        for index in indices:
            if len(index) >= self.batch_size / 2:
                index = torch.LongTensor(index).to(device)
                result.append([states[index], actions[index], log_probs_old[index], returns[index], advantages[index]])
        return result
    
    def reset(self):
        """Reset all trajectories from memory."""
        self.memory = []
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
