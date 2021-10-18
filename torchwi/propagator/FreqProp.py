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
        vp = vel.data.numpy()
        self.lvirt = -2*omega**2/vp**3
        mat = impedance_matrix_vpad(self.omega, vp, self.h, self.npml,mat='csr',dtype=self.dtype)
        self.solver.analyze(mat)
        self.solver.factorize()

#    def solve_impulse(self, sxs,sy,ry, amplitude=1.0):
#        isxs = (sxs/self.h).int() # source x position
#        self.isy = int(sy/self.h) # source y position
#        self.iry = int(ry/self.h) # receiver y position, used in surface_wavefield
#
#        nrhs = len(sxs)
#        f = np.zeros((nrhs,self.nxyp),dtype=self.dtype)
#        for ishot,isx in enumerate(isxs):
#            f[ishot, (isx+self.npml)*self.nyp + self.isy] = amplitude
#
#        u = self.solver.solve(f)
#        u.shape = (nrhs,self.nxp,self.nyp)
#        return self.cut_pml(u)

    def solve_forward(self, sxs,sy, amplitude=1.0):
        # distribute each source on two points (x only)
        isxs_left = (sxs/self.h).int() # source x position
        wgt_right = (sxs % self.h)/self.h # source weight
        wgt_left  = 1.0 - wgt_right

        self.isy = int(sy/self.h) # source y position

        nrhs = len(sxs)
        f = np.zeros((nrhs,self.nxyp),dtype=self.dtype)
        for ishot,isx_left in enumerate(isxs_left):
            # left
            f[ishot, (isx_left+self.npml)*self.nyp + self.isy] = wgt_left[ishot] * amplitude
            # right
            isx_right = isx_left+1
            if isx_right < self.nx:
                f[ishot, (isx_right+self.npml)*self.nyp + self.isy] = wgt_right[ishot] * amplitude

        u = self.solver.solve(f)
        u.shape = (nrhs,self.nxp,self.nyp)
        return self.cut_pml(u)

    def solve_resid(self, resid, ry):
        iry = int(ry/self.h) # receiver y position
        nrhs = resid.shape[0]
        f = np.zeros((nrhs, self.nxp,self.nyp),dtype=self.dtype)
        if self.npml == 0:
            f[:,:,iry] = resid[:,:]
        else:
            f[:,self.npml:-self.npml,iry] = resid[:,:]
        f.shape=(nrhs,self.nxyp)
        b = self.solver.solve_transposed(f)
        b.shape = (nrhs,self.nxp,self.nyp)
        return self.cut_pml(b)

    def surface_wavefield(self,u,ry):
        # u: output from solve_forward/solve_impulse
        # input u.shape = (nrhs,nx,ny)
        # ry: receiver y position, full-offset
        iry = int(ry/self.h) # receiver y position
        return u[:,:,iry]

    def virtual_source(self,u):
        return self.lvirt * u

    def cut_pml(self,u):
        if self.npml == 0:
            return u
        else:
            return u[:,self.npml:-self.npml,:-self.npml]

    def finalize(self):
        self.solver.clear()



