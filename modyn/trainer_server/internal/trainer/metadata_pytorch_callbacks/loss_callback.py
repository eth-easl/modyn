from typing import Any

import torch
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.base_callback import BaseCallback
from modyn.trainer_server.internal.utils.metric_type import MetricType


class LossCallback(BaseCallback):
    def __init__(
        self, metadata_collector: MetadataCollector, loss_criterion_func: Any, loss_criterion_args: dict
    ) -> None:
        super().__init__()
        self._metadata_collector = metadata_collector
        self._sum_train_loss = 0.0
        self._average_train_loss = 0.0
        self._loss_criterion = loss_criterion_func(**loss_criterion_args, reduction="none")

    def on_batch_before_update(
        self,
        model: torch.nn.Module,
        optimizer: dict[str, torch.optim.Optimizer],
        batch_number: int,
        sample_ids: list[str],
        data: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        reduced_loss: torch.Tensor,
    ) -> None:
        super().on_batch_before_update(model, optimizer, batch_number, sample_ids, data, target, output, reduced_loss)
        loss_per_sample = self._loss_criterion(output, target)
        self._sum_train_loss += loss_per_sample.sum()
        self._metadata_collector.add_per_sample_metadata_for_batch(
            MetricType.LOSS, list(sample_ids), loss_per_sample.tolist()
        )

    def on_train_end(
        self,
        model: torch.nn.Module,
        optimizer: dict[str, torch.optim.Optimizer],
        total_samples: int,
        total_batches: int,
    ) -> None:
        super().on_train_end(model, optimizer, total_samples, total_batches)
        self._average_train_loss = self._sum_train_loss / total_samples
        self._metadata_collector.add_per_trigger_metadata(MetricType.LOSS, self._average_train_loss)
