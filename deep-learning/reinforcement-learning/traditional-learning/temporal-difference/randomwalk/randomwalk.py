import sys
import numpy as np
from six import StringIO
from string import ascii_uppercase

from gym import utils
from gym.envs.toy_text import discrete

WEST, EAST = 0, 1

class RandomWalkEnv(discrete.DiscreteEnv):
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self, n_states=7, p_stay=0.0, p_backward=0.5):
        
        # two terminal states added
        self.shape = (1, n_states + 2)
        self.start_state_index = self.shape[1]//2
        
        self.nS = nS = np.prod(self.shape)
        self.nA = nA = 2
        
        P = {}
        for s in range(nS):
            P[s] = {}
            for a in range(nA):
                p_forward = 1.0 - p_stay - p_backward
                
                s_forward = np.clip(s - 1 if a == WEST else s + 1, 0, nS - 1) if s != 0 and s != nS - 1 else s
                s_backward = np.clip(s + 1 if a == WEST else s - 1, 0, nS - 1) if s != 0 and s != nS - 1 else s

                r_forward = 1.0 if s == nS - 2 and s_forward == nS - 1 else 0.0
                r_backward = 1.0 if s == nS - 2 and s_backward == nS - 1 else 0.0

                d_forward = s >= nS - 2 and s_forward == nS - 1 or s <= 1 and s_forward == 0
                d_backward = s >= nS - 2 and s_backward == nS - 1 or s <= 1 and s_backward == 0

                P[s][a] = [
                    (p_forward, s_forward, r_forward, d_forward),
                    (p_stay, s, 0.0, s == nS - 1 or s == 0),
                    (p_backward, s_backward, r_backward, d_backward)
                ]
                
        isd = np.zeros(nS)
        isd[self.start_state_index] = 1.0
        discrete.DiscreteEnv.__init__(self, nS, nA, P, isd)
        
    def render(self, mode='human', close=False):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = np.asarray(['[' + ascii_uppercase[:self.shape[1] - 2] + ']'], dtype='c').tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        color = 'red' if self.s == 0 else 'green' if self.s == self.nS - 1 else 'yellow'
        desc[0][self.s] = utils.colorize(desc[0][self.s], color, highlight=True)
        outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
            