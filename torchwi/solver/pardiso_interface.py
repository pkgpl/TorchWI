from ctypes import POINTER, c_int, c_longlong, byref
from ctypes import cdll
from numpy import ctypeslib
import sys

platform = sys.platform
libname = {'linux':'libmkl_rt.so',
        'darwin':'libmkl_rt.dylib',
        'win32':'mkl_rt.dll'}

MKLlib = cdll.LoadLibrary(libname[platform])

c_int_p = POINTER(c_int)
c_long_p = POINTER(c_longlong)

# pardisoinit
_pardisoinit = MKLlib.pardisoinit
_pardisoinit.argtypes = [c_long_p, # pt
                        c_int_p, # mtype
                        c_int_p] # iparm
_pardisoinit.restype = None

# pardiso
_pardiso = MKLlib.pardiso
_pardiso.argtypes = [c_long_p, # pt
                    c_int_p,      # maxfct
                    c_int_p,      # mnum
                    c_int_p,      # mtype
                    c_int_p,      # phase
                    c_int_p,      # n
                    POINTER(None),# a
                    c_int_p,      # ia
                    c_int_p,      # ja
                    c_int_p,      # perm
                    c_int_p,      # nrhs
                    c_int_p,      # iparm
                    c_int_p,      # msglvl
                    POINTER(None),# b
                    POINTER(None),# x
                    c_int_p]      # error)
_pardiso.restype = None

# pardiso_export
_pardiso_export = MKLlib.pardiso_export
_pardiso_export.argtypes = [c_long_p, # pt
                           POINTER(None),# values
                           c_int_p,      # rows
                           c_int_p,      # columns
                           c_int_p,      # step
                           c_int_p,      # iparm
                           c_int_p]      # error
_pardiso_export.restype = None


def pardisoinit(pt,mtype,iparm):
    _pardisoinit( pt.ctypes.data_as(c_long_p), # pt
            byref(c_int(mtype)),               # mtype
            iparm.ctypes.data_as(c_int_p) )    # iparm


def pardiso(pt,maxfct,mnum,mtype,phase,
        n,a,ia,ja, perm,nrhs, iparm,msglvl,
        b,x,error):
    ctypes_dtype = ctypeslib.ndpointer(a.dtype)
    _pardiso( pt.ctypes.data_as(c_long_p), # pt
            byref(c_int(maxfct)),  # maxfct
            byref(c_int(mnum)),    # mnum
            byref(c_int(mtype)),   # mtype
            byref(c_int(phase)),   # phase
            byref(c_int(n)),       # n
            a.ctypes.data_as(ctypes_dtype), # a
            ia.ctypes.data_as(c_int_p), # ia
            ja.ctypes.data_as(c_int_p), # ja
            perm.ctypes.data_as(c_int_p), # perm
            byref(c_int(nrhs)),         # nrhs
            iparm.ctypes.data_as(c_int_p), # iparm
            byref(c_int(msglvl)),  # msglvl
            b.ctypes.data_as(ctypes_dtype), # b
            x.ctypes.data_as(ctypes_dtype), # x
            byref(c_int(error)) ) # error


def pardiso_export(pt,values,rows,columns,step,iparm,error):
    ctypes_dtype = ctypeslib.ndpointer(values.dtype)
    _pardiso_export( pt.ctypes.data_as(c_ong_p),# pt
            values.ctypes.data_as(ctypes_dtype),# values
            rows.ctypes.data_as(c_int_p),       # rows
            columns.ctypes.data_as(c_int_p),    # columns
            byref(c_int(step)),                 # step
            iparm.ctypes.data_as(c_int_p),      # iparm
            byref(c_int(error)) )               # error

