import torch
import numpy as np

torch_zero = torch.tensor(0.,dtype=torch.float64)
log_tolmin = 1.e-100

class LaplLogResid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frd, true):
        #nrhs, nx = true.shape
        resid = torch.where(torch.abs(true) > log_tolmin, torch.log(torch.abs(frd/true)), torch_zero.to(frd.device))

        ctx.save_for_backward(frd)
        return resid

    @staticmethod
    def backward(ctx, grad_output):
        frd, = ctx.saved_tensors
        grad_input = torch.where(torch.abs(frd)>log_tolmin, grad_output/frd, torch_zero.to(frd.device))
        return grad_input, None


class LaplLogLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frd, true):
        #nrhs, nx = true.shape
        mask = (torch.abs(true)>log_tolmin) & (torch.abs(frd.data)>log_tolmin)
        resid = torch.where(mask, torch.log(torch.abs(frd.data/true)), torch_zero.to(frd.device))
        loss = 0.5*torch.sum(resid**2)
        ctx.save_for_backward(frd.data,resid,mask)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        frd,resid,mask = ctx.saved_tensors
        grad_input = torch.where(mask, resid/frd, torch_zero.to(frd.device))
        del frd
        del resid
        del mask
        return grad_input, None


class LaplLogLossResid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frd, true):
        #nrhs, nx = true.shape
        resid = torch.where(torch.abs(true)>log_tolmin, torch.log(torch.abs(frd/true)), torch_zero.to(frd.device))
        loss = 0.5*torch.sum(resid**2)
        ctx.save_for_backward(frd,resid)
        return loss, resid

    @staticmethod
    def backward(ctx, grad_output, grad_output_resid):
        frd,resid = ctx.saved_tensors
        grad_input = torch.where(torch.abs(frd)>log_tolmin, resid/frd, torch_zero.to(frd.device))
        return grad_input, None


