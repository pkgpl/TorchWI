import torch
import numpy as np


class SourceEstimationLaplLog():
    def __init__(self):
        self.amp = 1.0
        self.zero()

    def zero(self):
        self.isum = torch.tensor(0)
        self.sum_amp = torch.tensor(0.,dtype=torch.float64)

    def add(self, resid):
        self.sum_amp += torch.sum(resid).item()
        self.isum += torch.sum(resid != 0).item()

    def step(self):
        log_amp = np.log(self.amp)
        log_amp -= self.sum_amp/self.isum
        self.amp = np.exp(log_amp)
        self.zero()

    def amplitude(self):
        return self.amp


