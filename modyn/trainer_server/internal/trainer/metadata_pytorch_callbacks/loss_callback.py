import torch

from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.base_callback import BaseCallback


class LossCallback(BaseCallback):
    def __init__(self, metadata_collector, loss_criterion_func, loss_criterion_args) -> None:
        super().__init__()
        self._metadata_collector = metadata_collector
        self._average_train_loss = 0.0
        self._loss_criterion = loss_criterion_func(**loss_criterion_args, reduction='None')

    def on_batch_before_update(self, sample_ids, data, target, output, reduced_loss):
        loss_per_sample = self._loss_criterion(output, target)
        self._average_train_loss += reduced_loss.item() * data.shape[0]
        self._metadata_collector.add_per_sample_metadata_for_batch(
            'loss',
            sample_ids[:,0].tolist(),
            loss_per_sample.tolist()
        )

    def on_train_end(self, total_num_samples):
        self._average_train_loss /= total_num_samples
        self._metadata_collector.add_per_trigger_metadata('loss', self._average_train_loss)