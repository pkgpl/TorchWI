import torch
from .Freq import BaseSourceEstimation


class LaplLogSourceEstimation(BaseSourceEstimation):
    def __init__(self,amp=1.0,device='cpu',log_tolmin=1.e-100,dtype=torch.float64):
        super().__init__(amp,device,dtype)
        self.torch_zero = torch.tensor(0.,dtype=self.dtype,device=self.device)
        self.log_tolmin = log_tolmin

    @torch.no_grad()
    def zero(self):
        self.isum    = torch.tensor(0,device=self.device)
        self.sum_amp = torch.tensor(0.,dtype=self.dtype,device=self.device)

    @torch.no_grad()
    def add_resid(self, resid):
        self.sum_amp += torch.sum(resid.data)
        self.isum    += torch.sum(resid.data != 0)

    @torch.no_grad()
    def add(self, frd,true):
        mask = (torch.abs(true) > self.log_tolmin) & (torch.abs(frd.data) > self.log_tolmin)
        resid = torch.where(mask, torch.log(torch.abs(frd.data/true)), self.torch_zero)
        self.sum_amp += torch.sum(resid)
        self.isum += torch.sum(mask)

    @torch.no_grad()
    def step(self):
        log_amp = torch.log(self.amp)
        log_amp -= self.sum_amp/self.isum
        self.amp = torch.exp(log_amp)
        self.zero()

