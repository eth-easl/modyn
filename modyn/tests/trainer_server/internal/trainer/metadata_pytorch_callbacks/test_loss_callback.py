from unittest.mock import patch
import torch
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector

from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.loss_callback import LossCallback

def get_loss_callback():
    metadata_collector =  MetadataCollector(0, 1)
    return LossCallback(metadata_collector, torch.nn.MSELoss, {})

def test_init():
    loss_callback = get_loss_callback()
    assert loss_callback._average_train_loss == 0.0
    assert isinstance(loss_callback._loss_criterion, torch.nn.MSELoss)
    assert loss_callback._loss_criterion.reduction == 'None'

# @patch.object(MetadataCollector, "add_per_sample_metadata_for_batch", return_value=None)
# def test_on_batch_before_update(test_add_per_sample_metadata_for_batch):
    # loss_callback = get_loss_callback()

    # loss_callback.on_batch_before_update()
    # assert loss_callback._average_train_loss == 0.0
    # test_add_per_sample_metadata_for_batch.assert_called_once_with()

    # loss_callback.on_batch_before_update()
    # assert loss_callback._average_train_loss == 0.0
    # test_add_per_sample_metadata_for_batch.assert_called_once_with()

@patch.object(MetadataCollector, "add_per_trigger_metadata", return_value=None)
def test_on_train_end(test_add_per_trigger_metadata):
    loss_callback = get_loss_callback()
    loss_callback._average_train_loss = 10.0
    loss_callback.on_train_end(10)
    assert loss_callback._average_train_loss == 1.0
    test_add_per_trigger_metadata.assert_called_once_with('loss', 1.0)