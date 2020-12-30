import torch
import numpy as np

def ca2rt(carr):
    if carr.dtype == np.complex64:
        # complex64 ndarray to float32 tensor
        return torch.from_numpy(np.ascontiguousarray(carr).view(np.float32))
    elif carr.dtype == np.complex128:
        return torch.from_numpy(np.ascontiguousarray(carr).view(np.float64))
    else:
        raise NotImplementedError("input dtype not supported")


def rt2ca(rtensor):
    # float32 tensor to complex64 ndarray
    if rtensor.dtype == torch.float32:
        return rtensor.data.numpy().view(np.complex64)
    elif rtensor.dtype == torch.float64:
        return rtensor.data.numpy().view(np.complex128)
    else:
        raise NotImplementedError("input dtype not supported")
