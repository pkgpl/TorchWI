import torch
import numpy as np

# for cuda shared memory: BDIMX, BDIMY same to those defined in "td2d_cuda.h"
BDIMX = 16
BDIMY = 16

def get_operator(_propagator):

    class TimeOperator(torch.autograd.Function):

        @staticmethod
        def forward(ctx, vel, args):
            # batch size=1
            # input/output vel,grad: (nx,ny)
            # frd: (nt,nx)
            # virt: (nt,dimxy)
            # internal vpad: (dimy,dimx)
            sx,sy,ry,m = args

            _propagator.forward(m.frd, m.virt,
                    m.u1, m.u2, m.u3, m.vpad, m.w,
                    m.order, m.dimx, m.dimy, m.nt,
                    m.h, m.dt,
                    sx.item(),sy.item(),ry.item())
                    
            # save for gradient calculation
            ctx.model = m
            ctx.save_for_backward(m.virt,ry)
            return m.frd.view(m.nt,m.nx)[:,:m.nx_org]

        @staticmethod
        def backward(ctx, grad_output):
            # resid = grad_output: (nt,nx)
            # grad_input: (nx,ny)
            virt,ry = ctx.saved_tensors
            m = ctx.model

            resid = grad_output
            grad = _propagator.backward(virt,
                    m.u1, m.u2, m.u3, m.vpad, resid,
                    m.order, m.dimx, m.dimy, m.nt,
                    m.h, m.dt, ry.item())
            # grad shape=(dimx,) # x fast
            grad = grad.view(m.dimy,m.dimx)[m.ne:m.ne+m.ny_org,m.ne:m.ne+m.nx_org]
            # grad_input: (nx,ny)
            grad_input = grad.transpose(0,1)
            return grad_input, None

    return TimeOperator.apply


def get_forward_operator(_propagator):

    class TimeOperator(torch.autograd.Function):

        @staticmethod
        def forward(ctx, vel, args):
            # batch size=1
            # input/output vel,grad: (nx,ny)
            # frd: (nt,nx)
            # virt: (nt,dimxy)
            # internal vpad: (dimy,dimx)
            sx,sy,ry,m = args

            _propagator.forward(m.frd, 
                    m.u1, m.u2, m.u3, m.vpad, m.w,
                    m.order, m.dimx, m.dimy, m.nt,
                    m.h, m.dt,
                    sx.item(),sy.item(),ry.item())
                    
            return m.frd.view(m.nt,m.nx)[:,:m.nx_org]

        @staticmethod
        def backward(ctx, grad_output):
            return None, None

    return TimeOperator.apply


class Time2d(torch.nn.Module):
    def __init__(self,nx,ny,h,w,dt,order,device):
        """
        input/output vel,grad shape = (nx,ny) # y fast
        interval vpad shape = (dimy,dimx) # x fast
        """
        super(Time2d, self).__init__()

        self.device = device
        self.h = h
        self.w = w.to('cpu') # wavelet in cpu for both cpu/cuda
        self.nt = len(w)
        self.dt = dt

        self.order = order
        self.ne = order//2
        self.nx_org, self.ny_org = nx,ny

        if self.device == 'cuda':
            self.nx=dim_pad(self.nx_org,BDIMX)
            self.ny=dim_pad(self.ny_org,BDIMY)
            from torchwi.propagator import td2d_cuda
            self.time_modeling = get_operator(td2d_cuda)
        else:
            self.nx = self.nx_org
            self.ny = self.ny_org
            from torchwi.propagator import td2d_cpu
            self.time_modeling = get_operator(td2d_cpu)
        self.dimx = self.nx + self.order
        self.dimy = self.ny + self.order
        self.dimxy = self.dimx * self.dimy

        # work array for efficiency
        self.u1   = torch.zeros(self.dimxy, device=self.device)
        self.u2   = torch.zeros(self.dimxy, device=self.device)
        self.u3   = torch.zeros(self.dimxy, device=self.device)
        self.frd  = torch.zeros((self.nt*self.nx), device=self.device)
        self.virt = torch.zeros((self.nt*self.dimxy), device=self.device)

    def pad_vel(self,vel):
        pad_shape = (self.ne, self.nx-self.nx_org+self.ne, self.ne, self.ny-self.ny_org+self.ne)
        with torch.no_grad():
            self.vpad = torch.nn.ReplicationPad2d(pad_shape)(vel.transpose(0,1).view(1,1,self.ny_org,self.nx_org)).view(-1)

    def forward(self, vel, sx,sy,ry):
        # vel (nx,ny)
        # batchsize=1 to save memory for source wavefield
        self.pad_vel(vel)
        return self.time_modeling(vel, (sx,sy,ry,self))


# for gpu shared memory
def dim_pad(n,BDIM):
    """
    return smallest multiple of BDIM larger than n
    """
    mult=int((n+BDIM-1)/BDIM)
    return mult*BDIM


class Time2dForward(torch.nn.Module):
    def __init__(self,nx,ny,h,w,dt,order,device):
        """
        input/output vel,grad shape = (nx,ny) # y fast
        interval vpad shape = (dimy,dimx) # x fast
        """
        super().__init__()

        self.device = device
        self.h = h
        self.w = w.to('cpu') # wavelet in cpu for both cpu/cuda
        self.nt = len(w)
        self.dt = dt

        self.order = order
        self.ne = order//2
        self.nx_org, self.ny_org = nx,ny

        if self.device == 'cuda':
            self.nx=dim_pad(self.nx_org,BDIMX)
            self.ny=dim_pad(self.ny_org,BDIMY)
            from torchwi.propagator import td2d_forward_cuda
            self.time_modeling = get_forward_operator(td2d_forward_cuda)
        else:
            self.nx = self.nx_org
            self.ny = self.ny_org
            from torchwi.propagator import td2d_forward_cpu
            self.time_modeling = get_forward_operator(td2d_forward_cpu)
        self.dimx = self.nx + self.order
        self.dimy = self.ny + self.order
        self.dimxy = self.dimx * self.dimy

        # work array for efficiency
        self.u1   = torch.zeros(self.dimxy, device=self.device)
        self.u2   = torch.zeros(self.dimxy, device=self.device)
        self.u3   = torch.zeros(self.dimxy, device=self.device)
        self.frd  = torch.zeros((self.nt*self.nx), device=self.device)

    def pad_vel(self,vel):
        pad_shape = (self.ne, self.nx-self.nx_org+self.ne, self.ne, self.ny-self.ny_org+self.ne)
        with torch.no_grad():
            self.vpad = torch.nn.ReplicationPad2d(pad_shape)(vel.transpose(0,1).view(1,1,self.ny_org,self.nx_org)).view(-1)

    def forward(self, vel, sx,sy,ry):
        # vel (nx,ny)
        # batchsize=1 to save memory for source wavefield
        self.pad_vel(vel)
        return self.time_modeling(vel, (sx,sy,ry,self))
