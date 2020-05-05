import numpy as np
import gym
from gym import spaces, error, utils

class CustomEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=np.int)
        
    def reset(self):
        
        return 'Reset Environment.'
    
    def step(self):
        
        return 'Step Environment.'
    
    def render(self, mode='human', close=False):
        
        return 'Render Environment.'
