import torch
from modyn.model_storage.internal.storage_strategies.abstract_difference_operator import AbstractDifferenceOperator
from modyn.utils import reconstruct_tensor_from_bytes


class XorDifferenceOperator(AbstractDifferenceOperator):
    @staticmethod
    def calculate_difference(tensor: torch.Tensor, tensor_prev: torch.Tensor) -> bytes:
        bytes_curr = tensor.numpy().tobytes()
        bytes_prev = tensor_prev.numpy().tobytes()

        return bytes(a ^ b for (a, b) in zip(bytes_curr, bytes_prev))

    @staticmethod
    def restore(tensor_prev: torch.Tensor, buffer: bytes) -> torch.Tensor:
        prev_model_data = tensor_prev.numpy().tobytes()
        new_model_data = bytes(a ^ b for (a, b) in zip(prev_model_data, buffer))
        return reconstruct_tensor_from_bytes(tensor_prev, new_model_data)
