import torch
import numpy as np
import horovod.torch as hvd


class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, sxy, ftrue, with_index=False, data_type='bin', shape=None, transpose=False):
        super().__init__()
        self.ftrue = ftrue
        self.with_index=with_index
        self.sxy = sxy
        self.data_type = data_type
        self.shape = shape
        self.transpose = transpose

    def __len__(self):
        return len(self.sxy)

    def __getitem__(self, index):
        truedata = self.load_data(index)
        sx,sy = self.sxy[index]
        if self.with_index:
            return sx,sy, truedata, index
        else:
            return sx,sy, truedata

    def load_data(self, index):
        if self.data_type == 'bin' or self.data_type == 'binary':
            truedata = self.load_bin(index)
        elif self.data_type == 'npy':
            truedata = self.load_npy(index)
        elif self.data_type == 'su':
            truedata = self.load_su(index)
        if self.transpose:
            truedata = truedata.T
        return torch.from_numpy(truedata)

    def load_bin(self, index):
        truedata = np.fromfile("%s%04d.bin"%(self.ftrue, index),dtype=np.float32)
        if self.shape:
            truedata.shape = shape
        return truedata

    def load_npy(self, index):
        return np.load("%s%04d.npy"%(self.ftrue, index))

    def load_su(self, index):
        from pkrh.io import su
        return su.fromfile("%s%04d.su"%(self.ftrue, index), key='data')


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

def time_forward_distributed_dataloader(sxy,size=None,rank=None, **kwargs):
    dataset = TimeForwardDataset(sxy, **kwargs)
    return _time_dist_dataloader(dataset, size,rank)


def time_distributed_dataloader(ftrue,sxy,size=None,rank=None, **kwargs):
    dataset = TimeDataset(sxy, ftrue, with_index=True, **kwargs)
    return _time_dist_dataloader(dataset, size,rank)

