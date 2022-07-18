import torch
import numpy as np
from torchwi.utils import to_cupy, to_tensor


class Freq2d(torch.nn.Module):
    def __init__(self,nx,ny,h,npml=10,mtype=13,dtype=np.complex64,device='cpu'):
        super(Freq2d, self).__init__()
        self.h=h
        self.device = device
        if device == 'cpu':
            from torchwi.propagator.FreqProp import Frequency2dFDM as Prop
            self.prop = Prop(nx,ny,h,npml,mtype,dtype)
            print("freq 2d cpu")
        else: # cuda
            from torchwi.propagator.FreqPropGPU import Frequency2dFDMGPU as Prop
            self.prop = Prop(nx,ny,h,npml,dtype)
            print("freq 2d cuda")
        self.op = FreqOperator.apply

    def factorize(self, omega, vel):
        self.vel=vel
        self.omega=omega
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

        resid = torch.from_numpy(grad_output)

        if model.device == 'cpu':
            resid = grad_output.numpy()
        else:
            resid = to_cupy(grad_output)

        b = model.prop.solve_resid(resid,ry)
        grad_input = torch.sum(torch.real(virt*to_tensor(b)), dim=0)
        return grad_input, None

