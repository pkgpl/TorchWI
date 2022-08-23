import torch


class FreqL2Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frd, true):
        resid = frd - true # complex value
        l2 = torch.real(0.5*torch.sum(resid*torch.conj(resid)))
        ctx.save_for_backward(resid)
        return l2

    @staticmethod
    def backward(ctx, grad_output):
        resid, = ctx.saved_tensors
        grad_input = torch.conj(resid)
        return grad_input, None
