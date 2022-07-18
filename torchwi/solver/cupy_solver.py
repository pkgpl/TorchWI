import sys
import numpy as np
import cupy as cp
import scipy.sparse
import cupyx.scipy.sparse
import cupyx.scipy.sparse.linalg
import torch
from torchwi.utils import to_cupy


class CupySolver():
    def __init__(self):
        self.solver = None

    def initialize(self):
        pass

    def finalize(self):
        del self.solver

    def clear(self):
        self.finalize()

    def analyze(self,A):
        pass

    def factorize(self,A):
        """
        Factorize using cpu
        https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.factorized.html
        A: scipy (csr) matrix
        """
        if type(A) == cupyx.scipy.sparse.csr.csr_matrix:
            self.mat = A
        elif type(A) == scipy.sparse.csr.csr_matrix:
            self.mat = cp.sparse.csr_matrix(A)
        else:
            print("Wrong matrix type")
            sys.exit(1)
        self.solver = cp.sparse.linalg.factorized(self.mat)

    def solve(self, rhs, trans='N', nrhs_first = False):
        """
        Solve using gpu
        trans
          'N': A * x = rhs
          'T': A.T * x = rhs
          'H': A.conj().T * x = rhs
        nrhs_first
           True:  rhs.shape = (nrhs, n)
           False: rhs.shape = (n, nrhs)
        """
        self.rhs = to_cupy(rhs)

        if rhs.ndim == 2 and nrhs_first:
            # if input  rhs shape = (nrhs, n)
            # use rhs tranposed to solve: solve rhs shape = (n, nrhs)
            self.rhs = self.rhs.T

        x = self.solver(self.rhs, trans)
        # x.shape = (n,nrhs)

        if rhs.ndim == 2 and nrhs_first:
            # if input  rhs shape = (nrhs, n), return x with same shape
            x = x.T

        return x

