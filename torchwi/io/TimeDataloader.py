import torch
import numpy as np
import horovod.torch as hvd


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

