import numpy as np
from .ld2d_base_fdm_pml import impedance_matrix_vpad
from .FreqProp import Frequency2dFDM


class Laplace2dFDM(Frequency2dFDM):
    def __init__(self,nx,ny,h,npml=10,mtype=11,dtype=np.float64):
        super().__init__(nx,ny,h,npml,mtype,dtype)

    def factorize(self, s, vel):
        self.s = -np.abs(s) # negative damping
        vp = vel.data.numpy()
        self.lvirt = 2*s**2/vp**3
        mat = impedance_matrix_vpad(self.s, vp, self.h, self.npml)
        self.solver.analyze(mat)
        self.solver.factorize()

