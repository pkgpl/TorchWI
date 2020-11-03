import numpy as np
from numba import jit


def pml_pad(vel, npml):
    return np.pad(vel, ((npml, npml), (0, npml)), mode='edge')

def pml_cut(vel, npml):
    return vel[npml:-npml,0:-npml]

@jit
def pml_damp(npml, v, h):
    nx, ny = v.shape
    dampx = np.zeros(nx)
    dampy = np.zeros(ny)
    thickness = npml * h
    for i in range(npml):
        xx = (npml - i) * h
        dampx[i] = 1.5 * v[npml, 0] / thickness * np.log(1000.) * (xx / thickness) ** 2
        dampx[-1 - i] = dampx[i]
        dampy[-1 - i] = dampx[i]
    return dampx, dampy
