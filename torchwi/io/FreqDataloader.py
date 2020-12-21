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


