from typing import Any

import torch
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.base_callback import BaseCallback


class LossCallback(BaseCallback):
    def __init__(
        self, metadata_collector: MetadataCollector, loss_criterion_func: Any, loss_criterion_args: dict
    ) -> None:
        super().__init__()
        self._metadata_collector = metadata_collector
        self._average_train_loss = 0.0
        self._loss_criterion = loss_criterion_func(**loss_criterion_args, reduction="none")

    # pylint:disable=arguments-differ
    def on_batch_before_update(
        self,
        sample_ids: list[str],
        data: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        reduced_loss: torch.Tensor,
    ) -> None:
        super().on_batch_before_update()
        loss_per_sample = self._loss_criterion(output, target)
        self._average_train_loss += reduced_loss.item() * data.shape[0]
        self._metadata_collector.add_per_sample_metadata_for_batch("loss", list(sample_ids), loss_per_sample.tolist())

    def on_train_end(self, total_num_samples: int) -> None:  # pylint:disable=arguments-differ
        super().on_train_end()
        self._average_train_loss /= total_num_samples
        self._metadata_collector.add_per_trigger_metadata("loss", self._average_train_loss)
