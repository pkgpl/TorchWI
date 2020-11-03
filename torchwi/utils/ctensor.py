import torch
import numpy as np

def ca2rt(carr):
    # complex64 ndarray to float32 tensor
    return torch.from_numpy(np.ascontiguousarray(carr).view(np.float32))

def rt2ca(rtensor):
    # float32 tensor to complex64 ndarray
    return rtensor.data.numpy().view(np.complex64)


