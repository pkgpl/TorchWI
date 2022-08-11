import torch
import numpy as np

# time

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
        truedata = np.fromfile(self.ftrue%index,dtype=np.float32)
        if self.shape:
            truedata.shape = shape
        return truedata

    def load_npy(self, index):
        return np.load(self.ftrue%index)

    def load_su(self, index):
        from pkrh.io import su
        return su.fromfile(self.ftrue%index, key='data')


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


# freq

def load_data(ftrue,nshot,dtype,device='cpu'):
    # truedata: traveltime (nshot, nx) for float
    # truedata: traveltime (nshot, 2*nx) for complex
    if dtype in [np.float32, np.float64]:
        truedata = torch.from_numpy(np.fromfile(ftrue,dtype=dtype))
    elif dtype in [np.complex64, np.complex128]:
        truedata = ca2rt(np.fromfile(ftrue,dtype=dtype))
    else:
        sys.stderr.write("Dataset: Wrong data type")
        sys.exit(1)
    return truedata.view(nshot,-1).to(device)


class FreqDataset(torch.utils.data.Dataset):
    def __init__(self, ftrue, sxy_all, dtype, device='cpu'):
        super().__init__()
        self.sxy_all = sxy_all.to(device)
        self.nshot = len(sxy_all)
        self.truedata = load_data(ftrue,self.nshot,dtype,device)

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


# lapl

class LaplDataset(FreqDataset):
    def __init__(self, ftrue, sxy_all, dtype=np.float64, device='cpu'):
        super().__init__(ftrue, sxy_all, dtype, device)


