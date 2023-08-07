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

    def reset(self):
        self.record_embedding = False
        self.embedding = None

class ModynModel(nn.Module, ABC):
    def __init__(self, record_embedding: bool = False) -> None:
        super().__init__()
        self.embedding_recorder = EmbeddingRecorder(record_embedding)

    @property
    def embedding(self) -> Optional[torch.Tensor]:
        assert self.embedding_recorder is not None
        return self.embedding_recorder.embedding

    @abstractmethod
    def get_last_layer(self) -> nn.Module:
        raise NotImplementedError()
