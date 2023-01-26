import torch

from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.base_callback import BaseCallback


class LossCallback(BaseCallback):
    def __init__(self, model, optimizer, loss_criterion_func, loss_criterion_args) -> None:
        super().__init__()
        self._average_train_loss = 0.0
        self._loss_criterion = loss_criterion_func(**loss_criterion_args, reduction='None')

    def on_batch_end(self, sample_ids, data, target, output, loss):
        loss_per_sample = self._loss_criterion(output, target)
        self._average_train_loss += loss.item() * data.shape[0]
        return sample_ids[:,0].tolist(), loss_per_sample.tolist()

    def on_train_end(self, total_num_samples):
        self._average_train_loss /= total_num_samples
        return self._average_train_loss