import torch
import numpy as np
import cupy as cp
from .FreqOperator import Freq2d
from torchwi.utils import to_cupy, to_tensor


class Tomo2d(Freq2d):
    def __init__(self,nx,ny,h,npml=0,mtype=13,dtype=np.complex128,device='cpu'):
        super(Tomo2d, self).__init__(nx,ny,h,npml,mtype,dtype,device)
        self.op = TomoOperator.apply

    def forward(self, sxs,sy,ry):
        return self.op(self.vel, (self, sxs,sy,ry))


def traveltime(u,omega_real):
    ttime = -torch.imag(torch.log(u))/omega_real
    return ttime.to(torch.float32)


class TomoOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vel, args):
        # nrhs: batch size
        # vel: (nx,ny)
        # u: (nrhs,nx,ny)
        # frd: (nrhs,nx) -> output frd: (nrhs, 2*nx) 2 for real and imaginary
        # virt: (nrhs,nx,ny)
        model, sxs,sy,ry = args # input: source x position, sy, ry, source amplitude

        u    = model.prop.solve_forward(sxs,sy)
        frd  = model.prop.surface_wavefield(u,ry)
        virt = model.prop.virtual_source(u)
        frd = to_tensor(frd)
        # save for gradient calculation
        ctx.model = model
        ctx.ry = ry
        ctx.save_for_backward(to_tensor(virt),frd)
        return traveltime(frd,model.omega.real)

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,nx), traveltime difference
        # b: (nrhs,nx,ny)
        virt,frd = ctx.saved_tensors
        model = ctx.model
        ry    = ctx.ry

        if model.device == 'cpu':
            resid = -model.prop.omega.real * grad_output.numpy() / frd
        else:
            resid = -model.prop.omega.real * to_cupy(grad_output) / frd

        b = model.prop.solve_resid(resid,ry)
        grad_input = torch.sum(torch.imag(virt*to_tensor(b)).to(torch.float32), dim=0)
        return grad_input, None
