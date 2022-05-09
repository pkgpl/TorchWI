import torch
import numpy as np
from .Dataset import TimeDataset, TimeForwardDataset
from .Dataset import FreqDataset
from .Dataset import AllFreqData

# time shots
def time_forward_dataloader(sxy, **kwargs):
    dataset = TimeForwardDataset(sxy, **kwargs)
    return torch.utils.data.DataLoader(dataset, batch_size=1)


def time_dataloader(ftrue,sxy, **kwargs):
    dataset = TimeDataset(sxy, ftrue, with_index=True, **kwargs)
    return torch.utils.data.DataLoader(dataset, batch_size=1)

# frequencies, damping constants
def freq_dataloader(freqs, ifreqs=None):
    if ifreqs is None:
        ifreqs = np.arange(len(freqs))
    dataset = AllFreqData(freqs, ifreqs)
    return torch.utils.data.DataLoader(dataset, batch_size=1)

def lapl_dataloader(freqs, ifreqs=None):
    return freq_dataloader(freqs,ifreqs)

# shots in the freq/lapl
def freq_shot_dataloader(ftrue,sxy_all,nrhs,dtype):
    dataset = FreqDataset(ftrue,sxy_all,dtype)
    dataloader =  torch.utils.data.DataLoader(dataset,batch_size=nrhs)
    return dataloader

def lapl_shot_dataloader(ftrue,sxy_all,nrhs,dtype=np.float64):
    return freq_shot_dataloader(ftrue,sxy_all,nrhs,dtype)
