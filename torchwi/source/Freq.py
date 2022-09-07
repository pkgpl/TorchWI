import torch

class BaseSourceEstimation():
    def __init__(self,amp,device='cpu',dtype=torch.complex128):
        self.device=torch.device(device)
        self.dtype=dtype
        self.amp=torch.tensor(amp,device=self.device,dtype=self.dtype)
        self.zero()

    def zero(self):
        pass

    def add(self, frd,true):
        pass

    def step(self):
        pass

    def amplitude(self):
        return self.amp


class FreqL2SourceEstimation(BaseSourceEstimation):
    def __init__(self,amp=1.0+0j,device='cpu',dtype=torch.complex128):
        super().__init__(amp,device,dtype)

    def zero(self):
        self.sumup = torch.tensor(0.,dtype=self.dtype,device=self.device)
        self.sumdn = torch.tensor(0.,dtype=self.dtype,device=self.device)

    @torch.no_grad()
    def add(self, frd,true):
        green = frd/self.amp
        self.sumup += torch.sum(torch.conj(green)*true)
        self.sumdn += torch.sum(torch.conj(green)*green)

    @torch.no_grad()
    def step(self):
        self.amp = self.sumup/self.sumdn
        self.zero()

