import torch
import numpy as np
from torchwi.utils.ctensor import ca2rt, rt2ca
from .FreqOperator import Freq2d


def traveltime(u,omega_real):
    return -np.imag(np.log(u))/omega_real


class Tomo2d(Freq2d):
    def __init__(self,nx,ny,h,npml=0,mtype=13,dtype=np.complex128):
        super(Tomo2d, self).__init__(nx,ny,h,npml,mtype,dtype)
        self.op = TomoOperator.apply

    def forward(self, sxs,sy,ry):
        return self.op(self.vel, (self, sxs,sy,ry))

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
        # save for gradient calculation
        ctx.model = model
        ctx.save_for_backward(ca2rt(virt),ca2rt(frd))
        return torch.from_numpy(traveltime(frd,model.omega.real))

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,nx), traveltime difference
        # b: (nrhs,nx,ny)
        virt,frd = ctx.saved_tensors
        model = ctx.model

        # float32 tensor to complex64 ndarray
        virt = rt2ca(virt)
        frd = rt2ca(frd)
        resid = -model.prop.omega.real * grad_output.numpy() / frd

        b = model.prop.solve_resid(resid)
        grad_input = torch.sum(torch.from_numpy(np.imag(virt*b)), dim=0)
        return grad_input, None


