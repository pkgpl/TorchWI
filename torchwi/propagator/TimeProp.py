import torch
import numpy as np
from . import td2d_base

def get_fdm(ne):
    if ne == 1:
        _fdm = td2d_base.fdm_o2
    elif ne == 2:
        _fdm = td2d_base.fdm_o4
    elif ne == 4:
        _fdm = td2d_base.fdm_o8
    else:
        raise NotImplementedError("Wrong order: use 2, 4, and 8")
    return _fdm


def forward_np(frd, virt, 
        u1, u2, u3, vpad, w,
        order, dimx ,dimy, nt,
        h, dt, sx, sy, ry):
    # frd: (nt*nx) -> (nt, nx)
    # virt: (nt*dimxy) -> (nt, dimxy)
    # u1,u2,u3: (dimxy,) -> (dimx, dimy)
    frd_org_shape = frd.shape
    frd.shape=(nt,dimx-order)
    virt.shape=(nt, dimx*dimy)
    u1.shape=(dimx,dimy)
    u2.shape=(dimx,dimy)
    u3.shape=(dimx,dimy)

    ne = int(order / 2)
    _fdm = get_fdm(ne)

    vp2 = vpad**2
    vp3 = (vpad**3).flatten()
    dt2= dt**2
    h2 = h**2
    dtoh2= dt2/h2
    hdt= h*dt

    isx = int(sx/h)
    isy = int(sy/h)
    iry = int(ry/h)

    nx = dimx-order
    ny = dimy-order
    if isx >= nx or isy >= ny:
        print("Wrong shot position: isx=%d, isy=%d"%(isx,isy))
        raise

    u1[:,:]=0.
    u2[:,:]=0.
    u3[:,:]=0.
    for it in range(nt):
        _fdm(u3, u2, u1, vp2, dtoh2, dimx, dimy)
        td2d_base.inject_source(u3, w[it], isx, isy, ne, vp2, dt2)
        td2d_base.bc_keys(u3, u2, u1, vpad, vp2, dt2, h2, hdt, ne, dimx, dimy)
        frd[it,:] = u3[ne:-ne, iry+ne]
        virt[it,:] = td2d_base.diff2(u3, u2, u1, dt2).flatten() / vp3
        u1, u2 ,u3 = u2, u3, u1
    frd.shape = frd_org_shape


def backward_np(virt,
        u1, u2, u3, vpad, resid,
        order, dimx, dimy, nt,
        h, dt, ry):
    # virt: (nt*dimxy) -> (nt, dimxy)
    # u1,u2,u3: (dimxy,) -> (dimx, dimy)
    # vpad: (dimx, dimy)
    # grad output: (dimxy,) -> (dimx, dimy)
    virt.shape=(nt, dimx*dimy)
    u1.shape=(dimx,dimy)
    u2.shape=(dimx,dimy)
    u3.shape=(dimx,dimy)

    ne = int(order / 2)
    _fdm = get_fdm(ne)

    vp2 = vpad**2
    dt2= dt**2
    h2 = h**2
    dtoh2= dt2/h2
    hdt= h*dt

    iry = int(ry/h)

    grad = np.zeros_like(vpad).flatten()
    u1[:,:]=0.
    u2[:,:]=0.
    u3[:,:]=0.
    for it in reversed(range(nt)):
        _fdm(u3, u2, u1, vp2, dtoh2, dimx, dimy)
        td2d_base.inject_sources(u3, resid[it,:], iry, ne, vp2, dt2)
        td2d_base.bc_keys(u3, u2, u1, vpad, vp2, dt2, h2, hdt, ne, dimx, dimy)
        grad[:] = grad[:] + -virt[it,:] * u3[:,:].flatten()
        u1, u2 ,u3 = u2, u3, u1
    grad.shape=(dimx,dimy)
    return grad


def forward(frd, virt, 
        u1, u2, u3, vpad, w,
        order, dimx ,dimy, nt,
        h, dt, sx, sy, ry):
    # frd: (nt * nx)
    # virt: (nt * dimxy)
    # u1,u2,u3: (dimxy,)
    # vpad: (dimy, dimx)
    return forward_np(frd.numpy(), virt.numpy(),
            u1.numpy(), u2.numpy(), u3.numpy(), vpad.numpy().reshape((dimy,dimx)).T, w.numpy(),
            order, dimx, dimy, nt,
            h, dt, sx, sy, ry)

def backward(virt,
        u1, u2, u3, vpad, resid,
        order, dimx, dimy, nt,
        h, dt, ry):
    # virt: (nt * dimxy)
    # u1,u2,u3: (dimxy,)
    # vpad: (dimy, dimx)
    # resid: (nt, nx)
    grad_np = backward_np(virt.numpy(),
        u1.numpy(), u2.numpy(), u3.numpy(), vpad.numpy().reshape((dimy,dimx)).T, resid.numpy(),
        order, dimx, dimy, nt,
        h, dt, ry)
    return torch.from_numpy(grad_np.T)

