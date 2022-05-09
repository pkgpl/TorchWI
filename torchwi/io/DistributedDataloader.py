import torch
import horovod.torch as hvd

# Time shots
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


# frequencies
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


# Laplace damping constants
def lapl_distributed_dataloader(freqs,ifreqs=None, size=None, rank=None):
    return freq_distributed_dataloader(freqs,ifreqs)

