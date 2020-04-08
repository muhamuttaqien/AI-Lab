import numpy as np


class RunningStat(object):
    
    def __init__(self, shape):
        
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
        
    def push(self, x):
        
        x = np.asarray(x)
        
        assert x.shape == self._M.shape
        
        self._n += 1
        
        if self._n == 1:
            self._M[...] = x
        else:
            old_M = self._M.copy()
            self._M[...] = old_M + (x - old_M) / self._n
            self._S[...] = self._S + (x - old_M) * (x - self._M)
            
    @property
    def n(self):
        return self._n
    
    @n.setter
    def n(self, n):
        self._n = n
        
    @property
    def mean(self):
        return self._M
    
    @mean.setter
    def mean(self, M):
        self._M = M
        
    @property
    def sum_square(self):
        return self._S
    
    @sum_square.setter
    def sum_square(self, S):
        self._S = S
        
    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)
    
    @property
    def std(self):
        return np.sqrt(self.var)
    
    @property
    def shape(self):
        return self._M.shape
        
class ZFilter():
    
    def __init__(self, shape, de_mean=True, de_std=True, clip=10.0):
        
        self.de_mean = de_mean
        self.de_std = de_std
        self.clip = clip
        
        self.rs = RunningStat(shape)
        
    def __call__(self, x, update=True):
        
        if update: self.rs.push(x)
            
        if self.de_mean:
            x = x - self.rs.mean
            
        if self.de_std:
            x = x / (self.rs.std + 1e-8)
        
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
    
        return x
    
    
    