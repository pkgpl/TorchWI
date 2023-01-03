import torch
from torchwi.propagator.FreqProp import Frequency2dFDM as Prop


class Freq2d(torch.nn.Module):
    @torch.no_grad()
    def __init__(self,nx,ny,h,npml=10,mtype=13,dtype='complex64',device='cpu'):
        super(Freq2d, self).__init__()
        self.h=h
        self.device = device
        self.prop = Prop(nx,ny,h,npml,mtype,dtype,device)
        self.op = FreqOperator.apply
        self.factorized = False

    @torch.no_grad()
    def factorize(self, omega, vel):
        #self.vel=vel
        self.omega=omega
        self.prop.factorize(omega,vel.data)
        self.factorized = True

    def forward(self, vel,sxs,sy,ry,amplitude=1.0):
    #def forward(self, sxs,sy,ry,amplitude=1.0):
        if self.factorized:
            return self.op(vel, (self, sxs,sy,ry, amplitude))
            #return self.op(self.vel, (self, sxs,sy,ry, amplitude))
        else:
            raise Exception("Factorization required before forward modeling")

    @torch.no_grad()
    def finalize(self):
        self.prop.finalize()
        self.factorized = False
        #del self.vel
        del self.omega


class FreqOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vel, args):
        # nrhs: batch size
        # vel: (nx,ny)
        # frd: (nrhs,nx)
        model, sxs,sy,ry, amplitude = args # input: source x position, sy, ry, source amplitude

        model.prop.solve_forward(sxs,sy,amplitude)
        frd  = model.prop.surface_wavefield(ry)
        # save for gradient calculation
        ctx.model = model
        return frd

    @staticmethod
    def backward(ctx, grad_output):
        # resid = grad_output: (nrhs,nx)
        model = ctx.model

        model.prop.solve_resid(grad_output.data)
        grad_input = model.prop.gradient()

        del ctx.model
        return grad_input, None

