import numpy as np
from .fd2d_base_fdm_pml import impedance_matrix_vpad
from torchwi.solver.pardiso_solver import PardisoSolver


class Frequency2dFDM():
    def __init__(self,nx,ny,h,npml=10,mtype=13,dtype=np.complex64):
        self.nx, self.ny = nx,ny
        self.h = h
        self.dtype = dtype
        # pml pad
        self.npml=npml
        self.nxp = self.nx+2*npml
        self.nyp = self.ny+npml
        self.nxyp = self.nxp*self.nyp
        self.solver = PardisoSolver(mtype=mtype, dtype=self.dtype)

    def factorize(self, omega, vel):
        self.omega = omega
        self.vel = vel
        self.lvirt = -2*omega**2/vel**3
        vp = vel.data.numpy()
        mat = impedance_matrix_vpad(self.omega, vp, self.h, self.npml)
        self.solver.analyze(mat)
        self.solver.factorize()

    def solve_impulse(self, sxs,sy,ry, amplitude=1.0):
        isxs = (sxs/self.h).int() # source x position
        self.isy = int(sy/self.h) # source y position
        self.iry = int(ry/self.h) # receiver y position

        nrhs = len(sxs)
        f = np.zeros((nrhs,self.nxyp),dtype=self.dtype)
        for ishot,isx in enumerate(isxs):
            f[ishot, (isx+self.npml)*self.nyp + self.isy] = amplitude

        u = self.solver.solve(f)
        u.shape = (nrhs,self.nxp,self.nyp)
        return u[:,self.npml:-self.npml,:-self.npml]

    def surface_wavefield(self,u):
        # input u.shape = (nrhs,nx,ny)
        return u[:,:,self.iry]

    def virtual_source(self,u):
        return self.lvirt * u

    def solve_resid(self, resid):
        nrhs = resid.shape[0]
        f = np.zeros((nrhs, self.nxp,self.nyp),dtype=self.dtype)
        f[:,self.npml:-self.npml,self.iry] = resid[:,:]
        f.shape=(nrhs,self.nxyp)
        b = self.solver.solve_transposed(f)
        b.shape = (nrhs,self.nxp,self.nyp)
        return b[:,self.npml:-self.npml,:-self.npml]

    def finalize(self):
        self.solver.clear()

