import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from numba import jit
from .pml_base import pml_damp, pml_pad


def impedance_matrix_vpad(omega, vel, h, npml, mat='csr', Dtype=np.float64):
    return impedance_matrix(omega, pml_pad(vel, npml), h, npml, mat, Dtype)


def impedance_matrix(omega, vel, h, npml, mat='csr', Dtype=np.float64):
    nx, ny = vel.shape
    nxy = nx * ny
    dampx, dampy = pml_damp(npml, vel, h)
    row, col, val = assemble_coo(omega, vel, h, dampx, dampy, Dtype)
    if mat == 'csr':
        return csr_matrix((val, (row, col)), shape=(nxy, nxy))
    elif mat == 'csc':
        return csc_matrix((val, (row, col)), shape=(nxy, nxy))
    elif mat == 'coo':
        return coo_matrix((val, (row, col)), shape=(nxy, nxy))
    else:
        import sys
        print("Wrong matrix type")
        sys.exit(1)


@jit
def assemble_coo(omega, v, h, dampx, dampy, Dtype=np.float64):
    nx, ny = v.shape
    xix = 1. - dampx / omega
    xiy = 1. - dampy / omega
    oh2 = 1. / h ** 2
    omega2 = -omega ** 2

    lindex = nx * ny * 5
    row = np.zeros(lindex, dtype=np.int32)
    col = np.zeros_like(row)
    val = np.zeros(lindex, dtype=Dtype)

    ie = 0
    for ix in range(nx):
        xix2 = 1. / xix[ix] ** 2
        for iy in range(ny):
            xiy2 = 1. / xiy[iy] ** 2

            k = ix * ny + iy
            # center
            row[ie], col[ie] = k, k
            val[ie] = 2. * oh2 * xix2 + 2. * oh2 * xiy2 - omega2 / v[ix, iy] ** 2
            ie += 1
            # left
            if ix != 0:
                row[ie], col[ie] = k, k - ny
                if ix < nx - 1:
                    val[ie] = -oh2 * xix2 - 1. / xix[ix] ** 3 * (xix[ix + 1] - xix[ix - 1]) / 4. * oh2
                else:
                    val[ie] = -oh2 * xix2 - 1. / xix[ix] ** 3 * (xix[ix] - xix[ix - 1]) / 4. * oh2
                ie += 1
            # right
            if ix != nx - 1:
                row[ie], col[ie] = k, k + ny
                if ix > 0:
                    val[ie] = -oh2 * xix2 + 1. / xix[ix] ** 3 * (xix[ix + 1] - xix[ix - 1]) / 4. * oh2
                else:
                    val[ie] = -oh2 * xix2 + 1. / xix[ix] ** 3 * (xix[ix + 1] - xix[ix]) / 4. * oh2
                ie += 1
            # top
            if iy != 0:
                row[ie], col[ie] = k, k - 1
                if iy < ny - 1:
                    val[ie] = -oh2 * xiy2 - 1. / xiy[iy] ** 3 * (xiy[iy + 1] - xiy[iy - 1]) / 4. * oh2
                else:
                    val[ie] = -oh2 * xiy2 - 1. / xiy[iy] ** 3 * (xiy[iy] - xiy[iy - 1]) / 4. * oh2
                ie += 1
            # bottom
            if iy != ny - 1:
                row[ie], col[ie] = k, k + 1
                if iy > 0:
                    val[ie] = -oh2 * xiy2 + 1. / xiy[iy] ** 3 * (xiy[iy + 1] - xiy[iy - 1]) / 4. * oh2
                else:
                    val[ie] = -oh2 * xiy2 + 1. / xiy[iy] ** 3 * (xiy[iy + 1] - xiy[iy]) / 4. * oh2
                ie += 1

    return row[:ie], col[:ie], val[:ie]
