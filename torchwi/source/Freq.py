import torch
import numpy as np
from torchwi.utils.ctensor import ca2rt, rt2ca

class BaseSourceEstimation():
    def __init__(self,amp):
        self.amp=amp
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
    def __init__(self,amp=1.0+0j):
        super().__init__(amp)

    def zero(self):
        self.sumup = 0.
        self.sumdn = 0.

    def add(self, frd,true):
        green = rt2ca(frd)/self.amp
        true = rt2ca(true)
        self.sumup += np.sum(np.conjugate(green)*true)
        self.sumdn += np.sum(np.conjugate(green)*green)

    def step(self):
        self.amp = self.sumup/self.sumdn
        self.zero()

