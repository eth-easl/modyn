from abc import ABC

import torch


class BaseCallback(ABC):
    def __init__(self) -> None:
        pass

    def on_train_begin(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        pass

    def on_train_end(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, total_samples: int, total_batches: int
    ) -> None:
        pass

    def on_batch_begin(
        self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, batch: torch.Tensor, batch_number: int
    ) -> None:
        pass

    def on_batch_before_update(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_number: int,
        sample_ids: list[str],
        data: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        reduced_loss: torch.Tensor,
    ) -> None:
        pass

    def on_batch_end(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        batch_number: int,
        sample_ids: list[str],
        data: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        reduced_loss: torch.Tensor,
    ) -> None:
        pass
