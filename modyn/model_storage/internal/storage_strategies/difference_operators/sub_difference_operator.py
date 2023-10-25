import torch
from modyn.model_storage.internal.storage_strategies.abstract_difference_operator import AbstractDifferenceOperator
from modyn.utils import reconstruct_tensor_from_bytes


class SubDifferenceOperator(AbstractDifferenceOperator):
    @staticmethod
    def calculate_difference(tensor: torch.Tensor, tensor_prev: torch.Tensor) -> bytes:
        diff = tensor - tensor_prev
        return diff.numpy().tobytes()

    @staticmethod
    def restore(tensor_prev: torch.Tensor, buffer: bytes) -> torch.Tensor:
        difference_tensor = reconstruct_tensor_from_bytes(tensor_prev, buffer)
        return tensor_prev + difference_tensor
