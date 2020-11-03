import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import os

src_dir="torchwi/propagator/td2d/"

ext_modules = [
        CppExtension("torchwi.propagator.td2d_cpu",[
            src_dir+'td2d_cpu.c',
            src_dir+'propagator_cpu.c',
            src_dir+"wrapper_cpu.cpp",
            ],
            extra_compile_args=["-fopenmp"])
        ]

if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension("torchwi.propagator.td2d_cuda",[
            src_dir+'td2d_cuda.cu',
            src_dir+'propagator_cuda.cu',
            src_dir+"wrapper_cuda.cpp",
            ])
    )

setuptools.setup(
        name='torchwi',
        version='0.3',
        description='Full waveform inversion using PyTorch',
        author='Wansoo Ha',
        author_email='wansooha@gmail.com',
        packages=setuptools.find_packages(),
        ext_modules=ext_modules,
        cmdclass={
            'build_ext': BuildExtension
        })
