import torch
import numpy as np
from torchwi.utils.ctensor import ca2rt, rt2ca


class FreqL2Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frd, true):
        # resid: (nrhs, 2*nx) 2 for real and imaginary
        resid = frd - true
        resid_c = rt2ca(resid)
        l2 = np.real(0.5*np.sum(resid_c*np.conjugate(resid_c)))
        ctx.save_for_backward(resid)
        return torch.tensor(l2)

    @staticmethod
    def backward(ctx, grad_output):
        resid, = ctx.saved_tensors
        grad_input = ca2rt(np.conjugate(rt2ca(resid)))
        return grad_input, None



