import torch
import numpy as np
from torchwi.utils import to_cupy, to_tensor


class Freq2d(torch.nn.Module):
    def __init__(self,nx,ny,h,npml=10,mtype=13,dtype=np.complex64,device='cpu'):
        super(Freq2d, self).__init__()
        self.h=h
        self.device = device
        from torchwi.propagator.FreqProp import Frequency2dFDM as Prop
        self.prop = Prop(nx,ny,h,npml,mtype,dtype,device)
        self.op = FreqOperator.apply
        self.factorized = False

    def factorize(self, omega, vel):
        self.vel=vel
        self.omega=omega
        self.prop.factorize(omega,vel)
        self.factorized = True

    def forward(self, sxs,sy,ry,amplitude=1.0):
        if self.factorized:
            return self.op(self.vel, (self, sxs,sy,ry, amplitude))
        else:
            raise Exception("Factorization required before forward modeling")

    def finalize(self):
        self.prop.solver.clear()
        self.factorized = False


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
        ctx.ry = ry
        ctx.save_for_backward(to_tensor(virt))
        return to_tensor(frd)

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,2*nx)
        # b: (nrhs,nx,ny)
        virt, = ctx.saved_tensors
        model = ctx.model
        ry    = ctx.ry

        if model.device == 'cpu':
            resid = grad_output.numpy()
        else:
            resid = to_cupy(grad_output)

        b = model.prop.solve_resid(resid,ry)
        grad_input = torch.sum(torch.real(virt*to_tensor(b)), dim=0)
        return grad_input, None

