import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import skfuzzy as fuzzy


class Fuzzy(object):
    
    def __init__(self, interval):
        
        assert len(interval) == 3, 'need in form of [min, max, span]'

        self.fuzzy = fuzzy
        
        self.x = np.arange(interval[0], interval[1], interval[2])
        self.fuzzy_set = {}
        self.n_fset = None
        
    def trinf(self, inputs, u_key):
        
        assert len(inputs) == 3, 'need in form of [min, max, min]'
        self.fuzzy_set[u_key] = self.fuzzy.trimf(self.x, inputs)
        
        return self.fuzzy.trimf(self.x, inputs)
    
    def trapmf(self, inputs, u_key):
        
        assert len(inputs) == 4, 'need in form of [min, max, max, min]'
        self.fuzzy_set[u_key] = self.fuzzy.trapmf(self.x, inputs)
        
        return self.fuzzy.trapmf(self.x, inputs)
    
    def gaussmf(self, inputs, u_key):
        
        assert len(inputs) == 2, 'need in form of [mean, sd]'
        self.fuzzy_set[u_key] = self.fuzzy.gaussmf(self.x, inputs[0], inputs[1])
        
        return self.fuzzy.gaussmf(self.x, inputs[0], inputs[1])
    
    def _plot(self, *args):
        
        c = ['g', 'm', 'b', 'k']
        
        plt.figure(figsize=(10, 5))
        for ids, i in enumerate(args):
            plt.plot(self.x, i, 'b', linewidth=1.5, color=c[ids])
        plt.show()
        
    def _interp_membership(self, val, inputs):
        
        assert len(val) == len(self.x), 'need in form of len(x)'
        inputs = int(inputs)
        
        return self.fuzzy.interp_membership(self.x, val, inputs)
    