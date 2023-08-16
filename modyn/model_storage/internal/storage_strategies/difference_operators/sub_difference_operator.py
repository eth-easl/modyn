from typing import BinaryIO

import torch
from modyn.model_storage.internal.storage_strategies.abstract_difference_operator import AbstractDifferenceOperator
from modyn.model_storage.internal.utils.data_types import read_tensor_from_bytes


class SubDifferenceOperator(AbstractDifferenceOperator):
    @staticmethod
    def calculate_difference(tensor: torch.Tensor, tensor_prev: torch.Tensor) -> bytes:
        diff = tensor - tensor_prev
        return diff.numpy().tobytes()

    @staticmethod
    def restore(tensor_prev: torch.Tensor, bytestream: BinaryIO) -> torch.Tensor:
        difference_tensor = read_tensor_from_bytes(tensor_prev, bytestream)
        return tensor_prev + difference_tensor
