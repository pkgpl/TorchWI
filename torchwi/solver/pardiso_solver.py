import numpy as np
import scipy.sparse as sp
from numpy import ctypeslib

from .pardiso_interface import pardisoinit, pardiso

"""
mtype options
  1 -> real and structurally symmetric
  2 -> real and symmetric positive definite
 -2 -> real and symmetric indefinite
  3 -> complex and structurally symmetric
  4 -> complex and Hermitian positive definite
 -4 -> complex and Hermitian indefinite
  6 -> complex and symmetric
 11 -> real and nonsymmetric
 13 -> complex and nonsymmetric


phase options
 11 -> Analysis
 12 -> Analysis, numerical factorization
 13 -> Analysis, numerical factorization, solve, iterative refinement
 22 -> Numerical factorization
 23 -> Numerical factorization, solve, iterative refinement
 33 -> Solve, iterative refinement
331 -> like phase=33, but only forward substitution
332 -> like phase=33, but only diagonal substitution (if available)
333 -> like phase=33, but only backward substitution
  0 -> Release internal memory for L and U matrix number mnum
 -1 -> Release all internal memory for all matrices
"""

class PardisoSolver():
    """Wrapper class for Intel MKL Pardiso solver. """
    def __init__(self, mtype=11, dtype=np.float64, verbose=False, initialize=True):
        '''
        Parameters
        ----------
        mtype : int, optional
            flag specifying the matrix type. The possible types are:

            - 1 : real and structurally symmetric (not supported)
            - 2 : real and symmetric positive definite
            - -2 : real and symmetric indefinite
            - 3 : complex and structurally symmetric (not supported)
            - 4 : complex and Hermitian positive definite
            - -4 : complex and Hermitian indefinite
            - 6 : complex and symmetric
            - 11 : real and nonsymmetric (default)
            - 13 : complex and nonsymmetric
        verbose : bool, optional
            flag for verbose output. Default is False.

        Returns
        -------
        None

        '''
        self.mtype = mtype
        self.dtype = dtype
        self.ctypes_dtype = ctypeslib.ndpointer(self.dtype)

        self.maxfct = 1
        self.mnum = 1
        self.msglvl = 1 if verbose else 0

        if initialize:
            self.initialize()


    def initialize(self):
        # Initialize pardiso
        self.pt = np.zeros(64, np.int64)
        self.iparm = np.zeros(64, dtype=np.int32)
        pardisoinit(self.pt, self.mtype, self.iparm)

        self.iparm[1] = 3 # Use parallel nested dissection for reordering
        self.iparm[23] = 1 # Use parallel factorization
        self.iparm[34] = 1 # Zero base indexing
        self.iparm[27] = 0 # double precision
        if self.dtype == np.float32  or self.dtype == np.complex64:
            self.iparm[27] = 1 # single precision


    def finalize(self):
        '''
        Clear the memory allocated from the solver.
        '''
        self.run_pardiso(phase=-1)

    def clear(self):
        self.finalize()


    def analyze(self,A):
        '''
        Parameters
        ----------
        A : scipy.sparse.csr.csr_matrix
            sparse matrix in csr format.
        '''
        # If A is symmetric, store only the upper triangular portion 
        if self.mtype in [2, -2, 4, -4, 6]:
            A = sp.triu(A, format='csr')
        elif self.mtype in [11, 13]:
            A = A.tocsr()
        if not A.has_sorted_indices:
            A.sort_indices()

        self.n = A.shape[0]
        self.a = A.data
        self.ia = A.indptr
        self.ja = A.indices
        self.perm = np.zeros(self.n,np.int32)

        out = self.run_pardiso(phase=11)


    def factorize(self,A=None):
        if A is not None:
            self.a = A.data
            self.ia = A.indptr
            self.ja = A.indices
        out = self.run_pardiso(phase=22)


    def solve(self, rhs):
        return self.run_pardiso(phase=33, rhs=rhs)

    def solve_transposed(self, rhs):
        if self.mtype in [3,4,-4,6,13]:
            self.iparm[11] = 2  # solve with conjugate transposed matrix A
        else:
            self.iparm[11] = 1  # solve with transposed matrix A
        x = self.solve(rhs)
        self.iparm[11] = 0
        return x


    def run_pardiso(self, phase, rhs=None):
        '''
        Run specified phase of the Pardiso solver.

        Parameters
        ----------
        phase : int
            Flag setting the analysis type of the solver:

            -  11 : Analysis
            -  12 : Analysis, numerical factorization
            -  13 : Analysis, numerical factorization, solve, iterative refinement
            -  22 : Numerical factorization
            -  23 : Numerical factorization, solve, iterative refinement
            -  33 : Solve, iterative refinement
            - 331 : like phase=33, but only forward substitution
            - 332 : like phase=33, but only diagonal substitution (if available)
            - 333 : like phase=33, but only backward substitution
            -   0 : Release internal memory for L and U matrix number mnum
            -  -1 : Release all internal memory for all matrices
        rhs : ndarray, optional
            Right hand side of the equation `A x = rhs`. Can either be a vector
            (array of dimension 1) or a matrix (array of dimension 2). Default
            is None.

        Returns
        -------
        x : ndarray
            Solution of the system `A x = rhs`, if `rhs` is provided. Is either
            a vector or a column matrix.

        '''

        if rhs is None:
            nrhs = 0
            x = np.zeros(1)
            rhs = np.zeros(1)
        else:
            if rhs.ndim == 1:
                nrhs = 1
            elif rhs.ndim == 2:
                nrhs = rhs.shape[0]
            else:
                msg = "Right hand side must either be a 1 or 2 dimensional "+\
                      "array. Higher order right hand sides are not supported."
                raise NotImplementedError(msg)
            vlen = nrhs*self.n
            rhs.shape=(vlen,)
            x = np.zeros(vlen, dtype=self.dtype)

        ERR = 0

        pardiso(self.pt,      # pt
                self.maxfct,  # maxfct
                self.mnum,    # mnum
                self.mtype,   # mtype
                phase,        # phase
                self.n,       # n
                self.a,       # a
                self.ia,      # ia
                self.ja,      # ja
                self.perm,    # perm
                nrhs,         # nrhs
                self.iparm,   # iparm
                self.msglvl,  # msglvl
                rhs,          # b
                x,            # x
                ERR)          # error

        if nrhs > 1:
            x.shape=(nrhs,self.n)
            rhs.shape=(nrhs,self.n)
        return x
