import sys
import numpy as np
import cupy as cp
import scipy.sparse
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
from torchwi.utils import to_cupy, to_tensor
import torch

cupy_mempool = cp.get_default_memory_pool()
cupy_pmempool = cp.get_default_pinned_memory_pool()


class CupySolver():
    @torch.no_grad()
    def __init__(self):
        self._solver = None

    def initialize(self):
        pass

    @torch.no_grad()
    def finalize(self):
        #del self._solver
        #self._solver = None
        cupy_mempool.free_all_blocks()
        cupy_pmempool.free_all_blocks()
        #print("finalized solver mem used:",cupy_mempool.used_bytes()/1024**3)
        #print("finalized solver mem total:",cupy_mempool.total_bytes()/1024**3)

    def clear(self):
        self.finalize()

    def analyze(self,A):
        pass

    @torch.no_grad()
    def factorize(self,A):
        """
        Factorize using cpu
        https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.factorized.html
        A: scipy (csr) matrix
        """
        if type(A) == cupyx.scipy.sparse.csr_matrix:
        #if type(A) == cupyx.scipy.sparse.csr.csr_matrix:
            _mat = A
        elif type(A) == scipy.sparse.csr.csr_matrix:
            _mat = cp.sparse.csr_matrix(A)
        else:
            print("Wrong matrix type")
            sys.exit(1)
        self._solver = cp.sparse.linalg.factorized(_mat)
        del _mat
        #print("solver fact mem used:",cupy_mempool.used_bytes()/1024**3)
        #print("solver fact mem total:",cupy_mempool.total_bytes()/1024**3)

    @torch.no_grad()
    def solve(self, rhs, trans='N', nrhs_first = False):
        """
        Solve using gpu
        trans
          'N': A * x = rhs
          'T': A.T * x = rhs
          'H': A.conj().T * x = rhs
        nrhs_first
           True:  rhs.shape = (nrhs, n)
           False(default): rhs.shape = (n, nrhs)
        """
        _rhs = to_cupy(rhs)

        if rhs.ndim == 2 and nrhs_first:
            # if input  rhs shape = (nrhs, n)
            # use rhs tranposed to solve: solve rhs shape = (n, nrhs)
            _rhs = _rhs.T

        x = self._solver(_rhs, trans)
        # x.shape = (n,nrhs)

        if rhs.ndim == 2 and nrhs_first:
            # if input  rhs shape = (nrhs, n), return x with same shape
            x = x.T

        del _rhs
        #print("solver solve mem used:",cupy_mempool.used_bytes()/1024**3)
        #print("solver solve mem total:",cupy_mempool.total_bytes()/1024**3)
        return to_tensor(x)

