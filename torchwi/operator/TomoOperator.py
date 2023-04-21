import torch
import numpy as np
from .FreqOperator import Freq2d
from torchwi.utils import to_tensor


def traveltime(u,omega_real):
    ttime = -torch.imag(torch.log(u))/omega_real
    return ttime.to(torch.float32)


class Tomo2d(Freq2d):
    def __init__(self,nx,ny,h,npml=0,mtype=13,dtype=np.complex128,device='cpu'):
        super(Tomo2d, self).__init__(nx,ny,h,npml,mtype,dtype,device)
        self.op = TomoOperator.apply

    def forward(self, vel, sxs,sy,ry):
        return self.op(vel, (self, sxs,sy,ry))


class TomoOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vel, args):
        # nrhs: batch size
        # vel: (nx,ny)
        # u: (nrhs,nx,ny)
        # frd: (nrhs,nx) -> output frd: (nrhs, 2*nx) 2 for real and imaginary
        # virt: (nrhs,nx,ny)
        model, sxs,sy,ry = args # input: source x position, sy, ry, source amplitude

        model.prop.solve_forward(sxs,sy)
        #u    = model.prop.solve_forward(sxs,sy)
        frd  = model.prop.surface_wavefield(ry)
        #frd  = model.prop.surface_wavefield(u,ry)
        #virt = model.prop.virtual_source(u)
        #virt = model.prop.lvirt * model.prop.cut_pml(model.prop.u)

        ###frd = to_tensor(frd)
        # save for gradient calculation
        ctx.model = model
        ctx.ry = ry
        ctx.save_for_backward(frd)
        #ctx.save_for_backward(to_tensor(virt),frd)
        ttime = traveltime(frd,model.omega.real)
        #print('vel',vel.dtype)
        #print('ttime',ttime.dtype)
        return ttime

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,nx), traveltime difference
        # b: (nrhs,nx,ny)
        #virt,frd = ctx.saved_tensors
        frd, = ctx.saved_tensors
        model = ctx.model
        ry    = ctx.ry

        resid = -model.omega.real * grad_output / frd


        model.prop.solve_resid(resid)
        #b = model.prop.solve_resid(resid)
        grad_input = model.prop.gradient('imag')
        #grad_input = torch.sum(torch.imag(virt*to_tensor(b)).to(torch.float32), dim=0)
        #print('grad_input',grad_input.dtype)
        return grad_input, None
