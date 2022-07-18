import torch
import numpy as np
from .FreqOperator import Freq2d
from torchwi.utils import to_cupy, to_tensor


class Lapl2d(Freq2d):
    def __init__(self,nx,ny,h,npml=10,mtype=11,dtype=np.float64,device='cpu'):
        super(Lapl2d, self).__init__(nx,ny,h,npml,mtype,dtype,device)
        if device=='cpu':
            from torchwi.propagator.LaplProp import Laplace2dFDM as Prop
            self.prop = Prop(nx,ny,h,npml,mtype,dtype)
        else: # cuda
            from torchwi.propagator.LaplPropGPU import Laplace2dFDMGPU as Prop
            self.prop = Prop(nx,ny,h,npml,mtype,dtype)
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
        ctx.ry = ry
        ctx.save_for_backward(torch.from_numpy(virt))
        return torch.from_numpy(frd)

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,nx)
        # b: (nrhs,nx,ny)
        virt, = ctx.saved_tensors
        model = ctx.model
        ry    = ctx.ry

        resid = grad_output

        b = model.prop.solve_resid(resid,ry)
        grad_input = torch.sum(virt*to_tensor(b), dim=0)
        return grad_input, None

