class BasicBuffer(object):
    
    def __init__(self):
        
        self.states = []; self.actions = []; self.rewards = []; self.dones = []
        self.log_probs = []
        
    def clear_memory(self):
        del self.states[:]; del self.actions[:]; del self.rewards[:]; del self.dones[:]
        del self.log_probs[:]
        
    def __len__(self):
        return len(self.states)
    