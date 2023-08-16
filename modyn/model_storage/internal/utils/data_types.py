"""
This class provides useful functionalities for different data types and conversions between them.
"""
import math
from typing import BinaryIO

import numpy as np
import torch

torch_dtype_to_numpy_dict = {
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}

numpy_dtype_to_torch_dict = {value: key for (key, value) in torch_dtype_to_numpy_dict.items()}

torch_dtype_to_byte_size = {
    torch.uint8: 1,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.float16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.complex64: 8,
    torch.complex128: 16,
}


def read_tensor_from_bytes(tensor: torch.Tensor, bytestream: BinaryIO) -> torch.Tensor:
    shape = tensor.shape
    num_bytes = math.prod(shape) * torch_dtype_to_byte_size[tensor.dtype]
    byte_data = bytestream.read(num_bytes)
    np_dtype = np.dtype(torch_dtype_to_numpy_dict[tensor.dtype])
    return create_tensor(byte_data, np_dtype, shape)


def create_tensor(buffer: bytes, dtype: np.dtype, shape: torch.Size) -> torch.Tensor:
    dtype = dtype.newbyteorder("<")
    np_array = np.frombuffer(buffer, dtype=dtype)
    array_tensor = torch.tensor(np.array(np_array))
    return torch.reshape(array_tensor, shape)
