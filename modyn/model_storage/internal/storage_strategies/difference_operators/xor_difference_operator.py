import math
from typing import BinaryIO

import numpy as np
import torch
from modyn.model_storage.internal.storage_strategies.abstract_difference_operator import AbstractDifferenceOperator
from modyn.model_storage.internal.utils.data_types import (
    create_tensor,
    torch_dtype_to_byte_size,
    torch_dtype_to_numpy_dict,
)


class XorDifferenceOperator(AbstractDifferenceOperator):
    @staticmethod
    def calculate_difference(tensor: torch.Tensor, tensor_prev: torch.Tensor) -> bytes:
        bytes_curr = tensor.numpy().tobytes()
        bytes_prev = tensor_prev.numpy().tobytes()

        return bytes(a ^ b for (a, b) in zip(bytes_curr, bytes_prev))

    @staticmethod
    def restore(tensor_prev: torch.Tensor, bytestream: BinaryIO) -> torch.Tensor:
        shape = tensor_prev.shape
        num_bytes = math.prod(shape) * torch_dtype_to_byte_size[tensor_prev.dtype]
        byte_data: bytes = bytestream.read(num_bytes)
        prev_model_data = tensor_prev.numpy().tobytes()
        new_model_data = bytes(a ^ b for (a, b) in zip(byte_data, prev_model_data))
        np_dtype = np.dtype(torch_dtype_to_numpy_dict[tensor_prev.dtype])
        return create_tensor(new_model_data, np_dtype, shape)
