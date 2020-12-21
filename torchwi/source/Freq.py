import torch
import numpy as np
from torchwi.utils.ctensor import ca2rt, rt2ca


class SourceEstimationFreqL2():
    def __init__(self):
        self.amp=1.0+0j
        self.zero()

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

    def amplitude(self):
        return self.amp


