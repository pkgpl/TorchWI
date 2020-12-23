import torch
import numpy as np

torch_zero = torch.tensor(0.,dtype=torch.float64)
log_tolmin = 1.e-100

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

    def add_data(self, frd,true):
        mask = (torch.abs(true)>log_tolmin) & (torch.abs(frd)>log_tolmin)
        resid = torch.where(mask, torch.log(torch.abs(frd/true)), torch_zero)
        self.sum_amp += torch.sum(resid).item()
        self.isum += torch.sum(resid != 0).item()

    def step(self):
        log_amp = np.log(self.amp)
        log_amp -= self.sum_amp/self.isum
        self.amp = np.exp(log_amp)
        self.zero()

    def amplitude(self):
        return self.amp


LaplLogSourceEstimation = SourceEstimationLaplLog
