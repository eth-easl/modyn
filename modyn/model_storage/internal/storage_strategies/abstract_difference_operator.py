from abc import ABC, abstractmethod
from typing import BinaryIO

import torch


class AbstractDifferenceOperator(ABC):
    """
    This is the base class for all difference operators. These operators can be used to calculate the difference
    between two successive models in the pipeline and later be used in a incremental model storage strategy.
    """

    @staticmethod
    @abstractmethod
    def calculate_difference(tensor: torch.Tensor, tensor_prev: torch.Tensor) -> bytes:
        """
        Calculate the difference between two tensors.

        Args:
            tensor: the tensor representing some weights of the current model.
            tensor_prev: the tensor representing the same weights of the preceding model.

        Returns:
            bytes: the byte-level difference.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def restore(tensor_prev: torch.Tensor, bytestream: BinaryIO) -> torch.Tensor:
        """
        Restores a weight tensor.

        Args:
            tensor_prev: the tensor representing some weights of the preceding model.
            bytestream: difference bytes from which to restore the weights of the current model.

        Returns:
            tensor: the weight tensor of the current model.
        """
        raise NotImplementedError()
