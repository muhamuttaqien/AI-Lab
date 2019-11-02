import numpy as np
import torch

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    
    # Cosine annealing with restarts from AllenNLP
    
    def __init__(self, optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initalized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        
        if not self._initalized:
            self._initalized = True
            return self.base_lrs
        
        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart
        
        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi * 
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    )
                )
            ) for lr in self.base_lrs
        ]
        
        if self._cycle_counter % self._updated_cycle_len == 0:
            # adjust the cycle length
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step
            
        return lrs