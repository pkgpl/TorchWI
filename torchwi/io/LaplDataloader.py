import torch
import numpy as np
import horovod.torch as hvd
from .FreqDataloader import FreqDataset
from .FreqDataloader import freq_distributed_dataloader, freq_shot_dataloader


class LaplDataset(FreqDataset):
    def __init__(self, ftrue, sxy_all):
        super().__init__(ftrue, sxy_all, dtype=np.float64)


def lapl_distributed_dataloader(freqs, size=None, rank=None):
    return freq_distributed_dataloader(freqs)

def lapl_shot_dataloader(ftrue,sxy_all,nrhs,dtype=np.float64):
    return freq_shot_dataloader(ftrue,sxy_all,nrhs,dtype)
