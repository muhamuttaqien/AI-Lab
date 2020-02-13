import random
import numpy as np
from collections import defaultdict

class Agent:
    
    def __init__(self, nS=500, nA=6):
        """Initialize agent.

        Params
        ======
        - nS: number of states of the environment
        - nA: number of actions available to the agent
        """
        self.nS = nS
        self.nA = nA
        self.Q_table = np.zeros([nS, nA])
        
    def select_action(self, state, epsilon=0.1):
        """Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        
        # apply epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = np.random.choice(self.nA) # explore action space
        else:
            action = np.argmax(self.Q_table[state]) # exploit learned values
            
        return action
    
    def step(self, state, action, reward, next_state, done, alpha=0.3, gamma=0.6):
        """Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        current_value = self.Q_table[state][action]
        Qsa_next = np.max(self.Q_table[next_state])
        
        # apply Sarsamax or Q-learning update rule
        new_value = (1 - alpha) * current_value + (alpha * (reward + gamma * Qsa_next))
        self.Q_table[state][action] = new_value
        