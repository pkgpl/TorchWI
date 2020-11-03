import torch
import numpy as np
from torchwi.utils.ctensor import ca2rt
import horovod.torch as hvd


class FreqDataset(torch.utils.data.Dataset):
    def __init__(self, ftrue, sxy_all, dtype):
        super().__init__()
        self.sxy_all = sxy_all
        self.nshot = len(sxy_all)
        self.truedata = self.load_true(ftrue,dtype)

    def __len__(self):
        return self.nshot

    def __getitem__(self,index):
        return self.sxy_all[index], self.truedata[index]

    def load_true(self,ftrue,dtype):
        # truedata: traveltime (nshot, nx) for float
        # truedata: traveltime (nshot, 2*nx) for complex
        if dtype in [np.float32, np.float64]:
            truedata = torch.from_numpy(np.fromfile(ftrue,dtype=dtype))
        elif dtype in [np.complex64, np.complex128]:
            truedata = ca2rt(np.fromfile(ftrue,dtype=dtype))
        else:
            import sys
            sys.stderr.write("Dataset: Wrong data type")
            sys.exit(1)
        return truedata.view(self.nshot,-1)


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, sxy, ftrue, with_index=False):
        super().__init__()
        self.ftrue = ftrue
        self.with_index=with_index
        self.sxy = sxy

    def __len__(self):
        return len(self.sxy)

    def __getitem__(self, index):
        truedata = np.fromfile("%s%04d"%(self.ftrue, index),dtype=np.float32)
        truedata = torch.from_numpy(truedata)
        sx,sy = self.sxy[index]
        if self.with_index:
            return sx,sy, truedata, index
        else:
            return sx,sy, truedata


class TimeForwardDataset(torch.utils.data.Dataset):
    def __init__(self, sxy, with_index=True):
        super().__init__()
        self.with_index=with_index
        self.sxy = sxy
        self.nshot = len(sxy)

    def __len__(self):
        return self.nshot

    def __getitem__(self,index):
        sx,sy = self.sxy[index]
        if self.with_index:
            return sx,sy, index
        else:
            return sx,sy


def _time_dist_dataloader(dataset, size=None,rank=None):
    # batch_size=1 for time-domain fwi to save memory
    if size is None: size = hvd.size()
    if rank is None: rank = hvd.rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=size, rank=rank)
    return torch.utils.data.DataLoader(dataset,batch_size=1,sampler=train_sampler)

def time_forward_distributed_dataloader(sxy,size=None,rank=None):
    dataset = TimeForwardDataset(sxy)
    return _time_dist_dataloader(dataset, size,rank)


def time_distributed_dataloader(ftrue,sxy,size=None,rank=None):
    dataset = TimeDataset(sxy, ftrue, with_index=True)
    return _time_dist_dataloader(dataset, size,rank)

