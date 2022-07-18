import torch
import numpy as np
import cupy as cp


def to_cupy(a):
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    if type(a) == cp._core.core.ndarray:
        a_cp = a
    elif type(a) == torch.Tensor:
        with cp.cuda.Device(a.get_device()):
            a_cp = cp.asarray(a)
        assert a_cp.__cuda_array_interface__['data'][0] == a.__cuda_array_interface__['data'][0]
    else: # np.ndarray
        a_cp = cp.asarray(a)
    return a_cp


def to_tensor(a):
    if type(a) == torch.Tensor:
        a_t = a
    elif type(a) == cp._core.core.ndarray:
        a_t = torch.as_tensor(a, device='cuda:%d'%(a.device.id))
        assert a.__cuda_array_interface__['data'][0] == a_t.__cuda_array_interface__['data'][0]
    else: # np.ndarray
        a_t =  torch.from_numpy(a)
    return a_t


def to_numpy(a):
    if type(a) == np.ndarray:
        a_np = a
    elif type(a) == torch.Tensor:
        if a.requires_grad:
            a_np = a.detach().cpu().numpy()
        else:
            a_np = a.cpu().numpy()
    else: # cupy
        a_np = cp.asnumpy(a)
    return a_np

