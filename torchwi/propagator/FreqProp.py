import numpy as np
from .fd2d_base_fdm_pml import impedance_matrix_vpad
from torchwi.utils import to_tensor, to_numpy
import torch

np_to_torch_dtype_dict = {
    np.dtype('float32')    : torch.float32,
    np.dtype('float64')    : torch.float64,
    np.dtype('complex64')  : torch.complex64,
    np.dtype('complex128') : torch.complex128
}


class Frequency2dFDM():
    @torch.no_grad()
    def __init__(self,nx,ny,h,npml=10,mtype=13,dtype='complex64',device='cpu',mxrhs=None):
        self.nx, self.ny = nx,ny
        self.h = h
        self.dtype = np.dtype(dtype)
        self.device = device
        self._set_solver(mtype)
        # pml pad
        self.npml=npml
        self.nxp = self.nx+2*npml
        self.nyp = self.ny+npml
        self.nxyp = self.nxp*self.nyp
        self.iry = None
        # rhs
        # nrhs: batch size
        self.mxrhs = mxrhs
        self.f = None
        self.u = None
        self.b = None
        if self.mxrhs is not None:
            self._init_rhs()

    @torch.no_grad()
    def _init_rhs(self):
        self.rhs_shape = (self.mxrhs, self.nxyp)
        self.pad_shape = (self.mxrhs, self.nxp, self.nyp)
        self.shape = (self.mxrhs, self.nx, self.ny)

        self.f = torch.zeros(self.rhs_shape, dtype=np_to_torch_dtype_dict[self.dtype],device=self.device,requires_grad=False)
        self.u = torch.zeros(self.pad_shape,dtype=np_to_torch_dtype_dict[self.dtype],device=self.device,requires_grad=False)
        self.b = torch.zeros_like(self.u)

    @torch.no_grad()
    def finalize(self):
        self.solver.finalize()
        del self.lvirt
        del self.f
        del self.u
        del self.b
        self.mxrhs = None

    @torch.no_grad()
    def _set_solver(self,mtype):
        if self.device == 'cpu':
            from torchwi.solver.pardiso_solver import PardisoSolver
            self.solver = PardisoSolver(mtype=mtype, dtype=self.dtype)
        else:
            from torchwi.solver.cupy_solver import CupySolver
            self.solver = CupySolver()

    @torch.no_grad()
    def _set_lvirt(self,omega,vel):
        self.lvirt = -2*omega**2/vel**3

    @torch.no_grad()
    def factorize(self, omega, vel):
        mat = impedance_matrix_vpad(omega, to_numpy(vel), self.h, self.npml,mat='csr',dtype=self.dtype)
        self.solver.factorize(mat)
        self._set_lvirt(omega,vel.data)

    @torch.no_grad()
    def _set_impulse_source(self, sxs,sy,amplitude=1.0):
        # distribute each source on two points (x only)
        isxs_left = (sxs/self.h).int() # source x position
        wgt_right = (sxs % self.h)/self.h # source weight
        wgt_left  = 1.0 - wgt_right

        self.isy = int(sy/self.h) # source y position

        nrhs = len(sxs)
        if self.mxrhs is None:
            self.mxrhs = nrhs
            self._init_rhs()
        else:
            self.f[:,:]=0.
        for ishot,isx_left in enumerate(isxs_left):
            # left
            self.f[ishot, (isx_left+self.npml)*self.nyp + self.isy] = wgt_left[ishot] * amplitude
            # right
            isx_right = isx_left+1
            if isx_right < self.nx:
                self.f[ishot, (isx_right+self.npml)*self.nyp + self.isy] = wgt_right[ishot] * amplitude

    @torch.no_grad()
    def _set_adjoint_source(self, resid):
        self.f[:,:] = 0.
        self.f = self.f.view(self.pad_shape)
        if self.npml == 0:
            self.f[:self.nrhs,:,self.iry] = resid[:,:]
        else:
            self.f[:self.nrhs,self.npml:-self.npml,self.iry] = resid[:,:]
        self.f = self.f.view(self.rhs_shape)

    @torch.no_grad()
    def solve_forward(self, sxs,sy, amplitude=1.0):
        self.nrhs = len(sxs)
        self._set_impulse_source(sxs,sy,amplitude)

        u = self.solver.solve(self.f[:self.nrhs,:], trans='N', nrhs_first=True)
        #u.shape = (self.nrhs,self.nxp,self.nyp)
        self.u[:self.nrhs,:,:] = u.view((self.nrhs,self.nxp,self.nyp))
        #self.u[:self.nrhs,:,:] = self.cut_pml(u)
        #del u

    @torch.no_grad()
    def solve_resid(self, resid_t):
        self._set_adjoint_source(resid_t.data)
        b = self.solver.solve(self.f[:self.nrhs,:], trans='T', nrhs_first=True)
        #b.shape = (self.nrhs,self.nxp,self.nyp)
        self.b[:self.nrhs,:,:] = b.view((self.nrhs,self.nxp,self.nyp))
        #self.b[:self.nrhs,:,:] = self.cut_pml(b)
        #del b

    @torch.no_grad()
    def surface_wavefield(self,ry):
        # ry: receiver y position, full-offset
        self.iry = int(ry/self.h) # receiver y position
        if self.npml == 0:
            return self.u[:self.nrhs,:,self.iry]
        else:
            return self.u[:self.nrhs,self.npml:-self.npml,self.iry]
        #return to_tensor(self.u[:self.nrhs,:,self.iry])

    @torch.no_grad()
    def gradient(self,part='real'):
        ub = torch.sum(self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:], dim=0)
        if self.npml == 0:
            grad = self.lvirt * ub
        else:
            grad = self.lvirt * ub[self.npml:-self.npml,:-self.npml]
        if part.lower() == 'real':
            return torch.real(grad)
        elif part.lower()[:4] == 'imag':
            return torch.imag(grad)
        else:
            return grad

#        if self.npml == 0:
#            return torch.real(self.lvirt * torch.sum(self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:], dim=0))
#        else:
#            return torch.real(self.lvirt * torch.sum(self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:], dim=0)[self.npml:-self.npml,:-self.npml])
#        #return torch.sum(torch.real(self.lvirt * self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:]), dim=0)
#
#    @torch.no_grad()
#    def gradient_imag(self):
#        if self.npml == 0:
#            return torch.imag(self.lvirt * torch.sum(self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:], dim=0))
#        else:
#            return torch.imag(self.lvirt * torch.sum(self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:], dim=0)[self.npml:-self.npml,:-self.npml])
#
#    @torch.no_grad()
#    def gradient_cmplx(self):
#        if self.npml == 0:
#            return self.lvirt * torch.sum(self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:], dim=0)
#        else:
#            return self.lvirt * torch.sum(self.u[:self.nrhs,:,:] * self.b[:self.nrhs,:,:], dim=0)[self.npml:-self.npml,:-self.npml]

    @torch.no_grad()
    def cut_pml(self,u):
        if self.npml == 0:
            return to_tensor(u)
        else:
            return to_tensor(u[:,self.npml:-self.npml,:-self.npml])

