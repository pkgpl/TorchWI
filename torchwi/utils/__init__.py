import torch
import numpy as np
import cupy as cp


def to_cupy(a):
    # https://docs.cupy.dev/en/stable/user_guide/interoperability.html
    if type(a) == cp._core.core.ndarray:
        return a
    elif type(a) == torch.Tensor:
        with cp.cuda.Device(a.get_device()):
            a_cp = cp.asarray(a)
        assert a_cp.__cuda_array_interface__['data'][0] == a.__cuda_array_interface__['data'][0]
        return a_cp
    else: # np.ndarray
        return cp.asarray(a)


def to_tensor(a):
    if type(a) == torch.Tensor:
        return a
    elif type(a) == cp._core.core.ndarray:
        a_t = torch.as_tensor(a, device='cuda:%d'%(a.device.id))
        assert a.__cuda_array_interface__['data'][0] == a_t.__cuda_array_interface__['data'][0]
        return a_t
    else: # np.ndarray
        return torch.from_numpy(a)


def to_numpy(a):
    if type(a) == np.ndarray:
        return a
    elif type(a) == torch.Tensor:
        if a.requires_grad:
            a_np = a.detach().cpu().numpy()
        else:
            a_np = a.cpu().numpy()
        return a_np
    else: # cupy
        return cp.asnumpy(a)

