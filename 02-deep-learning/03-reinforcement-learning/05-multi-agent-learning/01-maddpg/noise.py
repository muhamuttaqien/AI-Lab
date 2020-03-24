import random
import copy
import numpy as np


class OUNoise(object):
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        
        self.size = size
        
        np.random.seed(seed)
        self.seed = random.seed(seed)
        
        self.mu = mu * np.ones(self.size)
        self.theta = theta
        self.sigma = sigma
        
        self.reset()
        
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample."""
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        
        return self.state
    