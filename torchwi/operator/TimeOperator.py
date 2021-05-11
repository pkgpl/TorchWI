import torch
from . import TimeFunction

# for cuda shared memory: BDIMX, BDIMY same to those defined in "td2d_cuda.h"
BDIMX = 16
BDIMY = 16


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
        else:
            self.nx = self.nx_org
            self.ny = self.ny_org

        self.dimx = self.nx + self.order
        self.dimy = self.ny + self.order
        self.dimxy = self.dimx * self.dimy

        self.time_modeling = TimeFunction.get_operator(self.device, forward_only=True)

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



class Time2d(Time2dForward):
    def __init__(self,nx,ny,h,w,dt,order,device, prop='ext',exa=False):
        """
        input/output vel,grad shape = (nx,ny) # y fast
        interval vpad shape = (dimy,dimx) # x fast
        """
        super(Time2d, self).__init__(nx,ny,h,w,dt,order,device)
        self.time_modeling = TimeFunction.get_operator(self.device, prop, exa)

        # work array for efficiency
        if exa:
            self.exa = torch.zeros(self.dimxy, device=self.device)
            self.iexa = torch.zeros(self.dimxy, dtype=torch.int16, device=self.device)
        else:
            self.virt = torch.zeros((self.nt*self.dimxy), device=self.device)

