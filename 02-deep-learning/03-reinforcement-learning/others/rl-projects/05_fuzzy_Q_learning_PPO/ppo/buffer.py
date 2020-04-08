import random

import numpy as np
from collections import deque


class ReplayBuffer(object):
    
    def __init__(self, capacity):
        
        self.data = deque(maxlen=capacity)
        self.memory = deque(maxlen=96)
        self.pointer = 0
        
    def add(self, state, action, reward, next_state, done):
        
        experience = (state, action, reward, next_state, done)
        self.data.append(experience)
        
        if self.pointer < len(self.data):
            self.pointer += 1
            
    def sample(self, batch, agents=1):
        
        if agents == 1:
            
            states = np.array([self.data[i][0] for i in batch])
            actions = np.array([self.data[i][1] for i in batch])
            next_states = np.array([self.data[i][3] for i in batch])
        else:
            
            states = []
            actions = []
            next_states = []
            
            for i_agent in range(agents):
                
                states.append(np.array([self.data[i][0][i_agent] for i in batch]))
                actions.append(np.array([self.data[i][1][i_agent] for i in batch]))
                next_states.append(np.array([self.data[i][3][i_agent] for i in batch]))
                
        rewards = np.array([self.data[i][2] for i in batch])
        dones = np.array([self.data[i][4] for i in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __str__(self):
        
        memory_state = ''
        
        for states, actions, rewards, next_states, dones in self.data:
            
            if isinstance(states, list):
                
                for i_state in states:
                    memory_state += f'{i_state.shape}'
                memory_state += ';'
            else:
                memory_state += f'{states.shape};'
                
        return memory_state
    