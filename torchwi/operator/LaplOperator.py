import torch
import numpy as np
from torchwi.propagator.LaplProp import Laplace2dFDM as Prop
from .FreqOperator import Freq2d


class Lapl2d(Freq2d):
    def __init__(self,nx,ny,h,npml=10):
        super(Lapl2d, self).__init__(nx,ny,h,npml)
        self.prop = Prop(nx,ny,h,npml)
        self.op = LaplOperator.apply


class LaplOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vel, args):
        # nrhs: batch size
        # vel: (nx,ny)
        # u: (nrhs,nx,ny)
        # frd: (nrhs,nx)
        # virt: (nrhs,nx,ny)
        model, sxs,sy,ry, amplitude = args # input: source x position, sy, ry, source amplitude

        u    = model.prop.solve_forward(sxs,sy,amplitude)
        frd  = model.prop.surface_wavefield(u,ry)
        virt = model.prop.virtual_source(u)
        # save for gradient calculation
        ctx.model = model
        ctx.save_for_backward(virt)
        return frd

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,nx)
        # b: (nrhs,nx,ny)
        virt, = ctx.saved_tensors
        model = ctx.model

        resid = grad_output

        b = model.prop.solve_resid(resid)
        grad_input = torch.sum(torch.from_numpy(virt*b), dim=0)
        return grad_input, None




class SourceEstimationLogLapl():
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


