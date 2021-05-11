import numpy as np
from numba import jit


@jit
def fdm_o2(up, uo, um, vel2, dtoh2, dimx, dimy):
    ne = 1
    for i in range(ne, dimx - ne):
        for j in range(ne, dimy - ne):
            up[i, j] = vel2[i, j] * dtoh2 * (-4. * uo[i, j]
                                             + uo[i - 1, j] + uo[i + 1, j] + uo[i, j - 1] + uo[i, j + 1]) \
                       + 2. * uo[i, j] - um[i, j]


@jit
def fdm_o4(up, uo, um, vel2, dtoh2, dimx, dimy):
    ne = 2
    for i in range(ne, dimx - ne):
        for j in range(ne, dimy - ne):
            up[i, j] = vel2[i, j] * dtoh2 * (-5. * uo[i, j]
                                             + 1.3333333333 * (
                                             uo[i - 1, j] + uo[i + 1, j] + uo[i, j - 1] + uo[i, j + 1])
                                             - 0.0833333333 * (
                                             uo[i - 2, j] + uo[i + 2, j] + uo[i, j - 2] + uo[i, j + 2])) \
                       + 2. * uo[i, j] - um[i, j]


@jit
def fdm_o8(up, uo, um, vel2, dtoh2, dimx, dimy):
    ne = 4
    for i in range(ne, dimx - ne):
        for j in range(ne, dimy - ne):
            up[i, j] = vel2[i, j] * dtoh2 * (-5.694444444444444 * uo[i, j]
                                             + 1.600000000000000000 * (
                                             uo[i - 1, j] + uo[i + 1, j] + uo[i, j - 1] + uo[i, j + 1])
                                             - 0.200000000000000000 * (
                                             uo[i - 2, j] + uo[i + 2, j] + uo[i, j - 2] + uo[i, j + 2])
                                             + 0.025396825396825397 * (
                                             uo[i - 3, j] + uo[i + 3, j] + uo[i, j - 3] + uo[i, j + 3])
                                             - 0.0017857142857142857 * (
                                             uo[i - 4, j] + uo[i + 4, j] + uo[i, j - 4] + uo[i, j + 4])) \
                       + 2. * uo[i, j] - um[i, j]


@jit
def inject_source(up, wit, isx, isy, ne, vel2, dt2):
    """
    wit: source wavelet, single time sample
    dt2: dt**2
    """
    up[isx + ne, isy + ne] += vel2[isx + ne, isy + ne] * dt2 * wit


@jit
def inject_sources(up, src, isy, ne, vel2, dt2):
    """
    wit: source wavelet, single time sample
    dt2: dt**2
    """
    up[ne:-ne, isy + ne] += vel2[ne:-ne, isy + ne] * dt2 * src[:]


# 2nd order Keys boundary condition
CT1 = np.cos(np.pi / 6.)
CT2 = np.cos(np.pi / 12.)
CM12 = CT1 * CT2
CP12 = CT1 + CT2


@jit
def bc_keys(up, uo, um, vel, vel2, dt2, h2, hdt, ne, dimx, dimy):
    # right
    for ie in range(ne):
        ix = dimx - ne + ie
        for iy in range(dimy):
            up[ix, iy] = -vel2[ix, iy] * dt2 / CM12 * (
                (uo[ix, iy] - 2. * uo[ix - 1, iy] + uo[ix - 2, iy]) / h2 +
                CP12 / (vel[ix, iy] * hdt) * ((uo[ix, iy] - uo[ix - 1, iy]) - (um[ix, iy] - um[ix - 1, iy]))) \
                         + 2. * uo[ix, iy] - um[ix, iy]
    # left
    for ie in range(ne):
        ix = ne - 1 - ie
        for iy in range(dimy):
            up[ix, iy] = -vel2[ix, iy] * dt2 / CM12 * (
                (uo[ix, iy] - 2. * uo[ix + 1, iy] + uo[ix + 2, iy]) / h2 +
                CP12 / (vel[ix, iy] * hdt) * ((uo[ix, iy] - uo[ix + 1, iy]) - (um[ix, iy] - um[ix + 1, iy]))) \
                         + 2. * uo[ix, iy] - um[ix, iy]
    # bottom
    for ie in range(ne):
        iy = dimy - ne + ie
        for ix in range(dimx):
            up[ix, iy] = -vel2[ix, iy] * dt2 / CM12 * (
                (uo[ix, iy] - 2. * uo[ix, iy - 1] + uo[ix, iy - 2]) / h2 +
                CP12 / (vel[ix, iy] * hdt) * ((uo[ix, iy] - uo[ix, iy - 1]) - (um[ix, iy] - um[ix, iy - 1]))) \
                         + 2. * uo[ix, iy] - um[ix, iy]


@jit
def diff2(up, uo, um, dt2):
    return (up - 2. * uo + um) / dt2
