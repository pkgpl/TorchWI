import sys
import torch
import numpy as np
from torchwi.utils.ctensor import ca2rt
import horovod.torch as hvd

def load_data(ftrue,nshot,dtype):
    # truedata: traveltime (nshot, nx) for float
    # truedata: traveltime (nshot, 2*nx) for complex
    if dtype in [np.float32, np.float64]:
        truedata = torch.from_numpy(np.fromfile(ftrue,dtype=dtype))
    elif dtype in [np.complex64, np.complex128]:
        truedata = ca2rt(np.fromfile(ftrue,dtype=dtype))
    else:
        sys.stderr.write("Dataset: Wrong data type")
        sys.exit(1)
    return truedata.view(nshot,-1)


class FreqDataset(torch.utils.data.Dataset):
    def __init__(self, ftrue, sxy_all, dtype):
        super().__init__()
        self.sxy_all = sxy_all
        self.nshot = len(sxy_all)
        self.truedata = load_data(ftrue,self.nshot,dtype)

    def __len__(self):
        return self.nshot

    def __getitem__(self,index):
        return self.sxy_all[index], self.truedata[index]


class AllFreqData(torch.utils.data.Dataset):
    def __init__(self, freqs, ifreqs):
        super().__init__()
        self.freqs = freqs
        self.ifreqs = ifreqs

    def __len__(self):
        return len(self.freqs)

    def __getitem__(self,index):
        return self.ifreqs[index], self.freqs[index]


def _freq_dist_dataloader(dataset, size=None,rank=None):
    # batch_size=1 for time-domain fwi to save memory
    if size is None: size = hvd.size()
    if rank is None: rank = hvd.rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=size, rank=rank)
    return torch.utils.data.DataLoader(dataset,batch_size=1,sampler=train_sampler)


def freq_distributed_dataloader(freqs, ifreqs=None, size=None, rank=None):
    if ifreqs is None:
        ifreqs = np.arange(len(freqs))
    dataset = AllFreqData(freqs, ifreqs)
    return _freq_dist_dataloader(dataset, size,rank)


def freq_shot_dataloader(ftrue,sxy_all,nrhs,dtype):
    dataset = FreqDataset(ftrue,sxy_all,dtype)
    dataloader =  torch.utils.data.DataLoader(dataset,batch_size=nrhs)
    return dataloader
