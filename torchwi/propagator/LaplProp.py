import numpy as np
from .ld2d_base_fdm_pml import impedance_matrix_vpad
from .FreqProp import Frequency2dFDM
from torchwi.utils import to_numpy
import torch


class Laplace2dFDM(Frequency2dFDM):
    def __init__(self,nx,ny,h,npml=10,mtype=11,dtype='float64',device='cpu'):
        super().__init__(nx,ny,h,npml,mtype,dtype,device)

    @torch.no_grad()
    def factorize(self, s, vel):
        self.s = -np.abs(s) # negative damping
        vp = to_numpy(vel.data)
        mat = impedance_matrix_vpad(self.s, vp, self.h, self.npml, mat='csr',dtype=self.dtype)
        self.solver.factorize(mat)
        self._set_lvirt(s,vel)

    @torch.no_grad()
    def _set_lvirt(self,s,vel):
        self.lvirt = 2*s**2/vel**3
