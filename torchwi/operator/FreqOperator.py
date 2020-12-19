import torch
import numpy as np
from torchwi.utils.ctensor import ca2rt, rt2ca
from torchwi.propagator.FreqProp import Frequency2dFDM as Prop


class Freq2d(torch.nn.Module):
    def __init__(self,nx,ny,h,npml=10):
        super(Freq2d, self).__init__()
        self.h=h
        self.prop = Prop(nx,ny,h,npml)
        self.op = FreqOperator.apply

    def factorize(self, omega, vel):
        self.vel=vel
        self.prop.factorize(omega,vel)

    def forward(self, sxs,sy,ry,amplitude=1.0):
        return self.op(self.vel, (self, sxs,sy,ry, amplitude))

    def finalize(self):
        self.prop.solver.clear()


class FreqOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vel, args):
        # nrhs: batch size
        # vel: (nx,ny)
        # u: (nrhs,nx,ny)
        # frd: (nrhs,nx) -> output frd: (nrhs, 2*nx) 2 for real and imaginary
        # virt: (nrhs,nx,ny)
        model, sxs,sy,ry, amplitude = args # input: source x position, sy, ry, source amplitude

        u    = model.prop.solve_forward(sxs,sy,amplitude)
        frd  = model.prop.surface_wavefield(u,ry)
        virt = model.prop.virtual_source(u)
        # save for gradient calculation
        ctx.model = model
        ctx.save_for_backward(ca2rt(virt))
        return ca2rt(frd)

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,2*nx)
        # b: (nrhs,nx,ny)
        virt, = ctx.saved_tensors
        model = ctx.model

        # float32 tensor to complex64 ndarray
        virt = rt2ca(virt)
        resid = rt2ca(grad_output)

        b = model.prop.solve_resid(resid)
        grad_input = torch.sum(torch.from_numpy(np.real(virt*b)), dim=0)
        return grad_input, None


class SourceEstimationL2():
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


