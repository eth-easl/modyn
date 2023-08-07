from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import nn


# acknowledgment: github.com/PatrickZH/DeepCore/
class EmbeddingRecorder(nn.Module):
    def __init__(self, record_embedding: bool = False):
        super().__init__()
        self.record_embedding = record_embedding
        self.embedding: Optional[torch.Tensor] = None

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.record_embedding:
            self.embedding = tensor
        return tensor

    def __enter__(self) -> None:
        self.record_embedding = True

    def __exit__(self, *args: Any) -> None:
        self.reset()

    def reset(self) -> None:
        self.record_embedding = False
        self.embedding = None


class CoresetMethodsSupport(nn.Module, ABC):
    """
    This class is used to support some Coreset Methods.
    Embeddings, here defined as the activation before the last layer, are often used to estimate the importance of
    a point. To implement this class correctly, it is necessary to
        - implement the get_last_layer method
        - modify the forward pass so that the last layer embedding is recorded. For example, in a simple network like
            x = self.fc1(input)
            x = self.fc2(x)
            output = self.fc3(x)
        it must be modified as follows
            x = self.fc1(input)
            x = self.fc2(x)
            x = self.embedding_recorder(x)
            output = self.fc3(x)
    """

    def __init__(self, record_embedding: bool = False) -> None:
        super().__init__()
        self.embedding_recorder = EmbeddingRecorder(record_embedding)

    @property
    def embedding(self) -> Optional[torch.Tensor]:
        assert self.embedding_recorder is not None
        return self.embedding_recorder.embedding

    @abstractmethod
    def get_last_layer(self) -> nn.Module:
        """
        Returns the last layer. Used for example to obtain the pre-layer and post-layer dimensions of tensors

        """
        raise NotImplementedError()
