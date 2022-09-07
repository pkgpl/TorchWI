import numpy as np
import cupy as cp
from .fd2d_base_fdm_pml import impedance_matrix_vpad
from torchwi.utils import to_cupy, to_numpy, to_tensor
import torch


class Frequency2dFDM():
    def __init__(self,nx,ny,h,npml=10,mtype=13,dtype=np.complex64,device='cpu'):
        self.nx, self.ny = nx,ny
        self.h = h
        self.dtype = dtype
        self.device = device
        self._set_solver(mtype)
        # pml pad
        self.npml=npml
        self.nxp = self.nx+2*npml
        self.nyp = self.ny+npml
        self.nxyp = self.nxp*self.nyp

    @torch.no_grad()
    def _set_solver(self,mtype):
        if self.device == 'cpu':
            from torchwi.solver.pardiso_solver import PardisoSolver
            self.solver = PardisoSolver(mtype=mtype, dtype=self.dtype)
        else:
            from torchwi.solver.cupy_solver import CupySolver
            self.solver = CupySolver()

    @torch.no_grad()
    def _rhs(self,shape):
        if self.device == 'cpu':
            f = np.zeros(shape,dtype=self.dtype)
        else:
            f = cp.zeros(shape,dtype=self.dtype)
        return f

    @torch.no_grad()
    def _set_lvirt(self,omega,vel):
        self.lvirt = -2*omega**2/vel**3

    @torch.no_grad()
    def factorize(self, omega, vel):
        self.omega = omega
        vp = to_numpy(vel.data)
        mat = impedance_matrix_vpad(self.omega, vp, self.h, self.npml,mat='csr',dtype=self.dtype)
        self.solver.factorize(mat)
        self._set_lvirt(omega,vel)

    @torch.no_grad()
    def _impulse_source(self, sxs,sy,amplitude=1.0):
        # distribute each source on two points (x only)
        isxs_left = (sxs/self.h).int() # source x position
        wgt_right = (sxs % self.h)/self.h # source weight
        wgt_left  = 1.0 - wgt_right

        self.isy = int(sy/self.h) # source y position

        nrhs = len(sxs)
        f = self._rhs((nrhs,self.nxyp))
        for ishot,isx_left in enumerate(isxs_left):
            # left
            f[ishot, (isx_left+self.npml)*self.nyp + self.isy] = wgt_left[ishot] * amplitude
            # right
            isx_right = isx_left+1
            if isx_right < self.nx:
                f[ishot, (isx_right+self.npml)*self.nyp + self.isy] = wgt_right[ishot] * amplitude
        return f

    @torch.no_grad()
    def _adjoint_source(self, resid, ry):
        iry = int(ry/self.h) # receiver y position
        nrhs = resid.shape[0]
        f = self._rhs((nrhs,self.nxp,self.nyp))
        if self.npml == 0:
            f[:,:,iry] = resid[:,:]
        else:
            f[:,self.npml:-self.npml,iry] = resid[:,:]
        f.shape=(nrhs,self.nxyp)
        return f

    @torch.no_grad()
    def solve_forward(self, sxs,sy, amplitude=1.0):
        f = self._impulse_source(sxs,sy,amplitude)
        nrhs = f.shape[0]

        u = self.solver.solve(f, trans='N', nrhs_first=True)
        u.shape = (nrhs,self.nxp,self.nyp)
        return self.cut_pml(u)

    @torch.no_grad()
    def solve_resid(self, resid_t, ry):
        if self.device == 'cpu':
            resid = to_numpy(resid_t)
        else:
            resid = to_cupy(resid_t)
        f = self._adjoint_source(resid, ry)
        nrhs = f.shape[0]
        b = self.solver.solve(f,trans='T', nrhs_first=True)
        b.shape = (nrhs,self.nxp,self.nyp)
        return self.cut_pml(b)

    @torch.no_grad()
    def surface_wavefield(self,u,ry):
        # u: output from solve_forward/solve_impulse
        # input u.shape = (nrhs,nx,ny)
        # ry: receiver y position, full-offset
        iry = int(ry/self.h) # receiver y position
        return to_tensor(u[:,:,iry])

    @torch.no_grad()
    def virtual_source(self,u):
        return self.lvirt * u

    @torch.no_grad()
    def cut_pml(self,u):
        if self.npml == 0:
            return to_tensor(u)
        else:
            return to_tensor(u[:,self.npml:-self.npml,:-self.npml])

    @torch.no_grad()
    def finalize(self):
        self.solver.clear()
        del self.lvirt



