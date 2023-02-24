# flake8: noqa
# mypy: ignore-errors
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

abspath = os.path.dirname(os.path.realpath(__file__))

setup(
    ext_modules=[
        CUDAExtension(
            name="cuda_ext.fused_embedding",
            sources=[
                os.path.join(abspath, "cuda_src/pytorch_embedding_ops.cpp"),
                os.path.join(abspath, "cuda_src/gather_gpu_fused_pytorch_impl.cu"),
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_70", "-gencode", "arch=compute_80,code=sm_80"]},
        ),
        CUDAExtension(
            name="cuda_ext.interaction_volta",
            sources=[
                os.path.join(abspath, "cuda_src/dot_based_interact_volta/pytorch_ops.cpp"),
                os.path.join(abspath, "cuda_src/dot_based_interact_volta/dot_based_interact_pytorch_types.cu"),
            ],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                    "-gencode",
                    "arch=compute_70,code=sm_70",
                ],
            },
        ),
        CUDAExtension(
            name="cuda_ext.interaction_ampere",
            sources=[
                os.path.join(abspath, "cuda_src/dot_based_interact_ampere/pytorch_ops.cpp"),
                os.path.join(abspath, "cuda_src/dot_based_interact_ampere/dot_based_interact_pytorch_types.cu"),
            ],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                    "-gencode",
                    "arch=compute_80,code=sm_80",
                ],
            },
        ),
        CUDAExtension(
            name="cuda_ext.sparse_gather",
            sources=[
                os.path.join(abspath, "cuda_src/sparse_gather/sparse_pytorch_ops.cpp"),
                os.path.join(abspath, "cuda_src/sparse_gather/gather_gpu.cu"),
            ],
            extra_compile_args={"cxx": [], "nvcc": ["-arch=sm_70", "-gencode", "arch=compute_80,code=sm_80"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
