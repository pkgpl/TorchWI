import torch
from torchwi.propagator.FreqProp import Frequency2dFDM as Prop


class Freq2d(torch.nn.Module):
    def __init__(self,nx,ny,h,npml=10,mtype=13,dtype='complex64',device='cpu'):
        super(Freq2d, self).__init__()
        self.h=h
        self.device = device
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

    @torch.no_grad()
    def finalize(self):
        self.prop.finalize()
        self.factorized = False
        del self.vel


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
        # save for gradient calculation
        ctx.model = model
        ctx.ry = ry
        ctx.save_for_backward(u)
        return frd

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,2*nx)
        # b: (nrhs,nx,ny)
        model = ctx.model
        ry    = ctx.ry
        u,    = ctx.saved_tensors
        virt = model.prop.virtual_source(u)

        resid = grad_output
        b = model.prop.solve_resid(resid,ry)

        if b.type() in ['torch.ComplexFloatTensor','torch.cuda.ComplexFloatTensor']: # Frequency domain
            grad_input = torch.sum(torch.real(virt*b), dim=0)
        else: # torch.FloatTensor, torch.cuda.FloatTensor # Laplace domain
            grad_input = torch.sum(virt*b, dim=0)

        del ctx.model
        del ctx.ry
        return grad_input, None

