import torch
import numpy as np
from .FreqOperator import Freq2d
from torchwi.propagator.LaplProp import Laplace2dFDM as Prop


class Lapl2d(Freq2d):
    def __init__(self,nx,ny,h,npml=10,mtype=11,dtype=np.float64,device='cpu'):
        super(Lapl2d, self).__init__(nx,ny,h,npml,mtype,dtype,device)
        self.prop = Prop(nx,ny,h,npml,mtype,dtype,device)

