from unittest.mock import patch

import torch

from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.loss_callback import LossCallback
from modyn.trainer_server.internal.utils.metric_type import MetricType


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, data):
        return data


def get_mocks():
    model = MockModel()
    optimizer = {"model": torch.optim.SGD(model.parameters(), lr=0.1)}
    return model, optimizer


def get_loss_callback():
    metadata_collector = MetadataCollector(0, 1)
    return LossCallback(metadata_collector, torch.nn.MSELoss, {})


def test_init():
    loss_callback = get_loss_callback()
    assert loss_callback._average_train_loss == 0.0
    assert isinstance(loss_callback._loss_criterion, torch.nn.MSELoss)
    assert loss_callback._loss_criterion.reduction == "none"


@patch.object(MetadataCollector, "add_per_sample_metadata_for_batch", return_value=None)
def test_on_batch_before_update(test_add_per_sample_metadata_for_batch):
    loss_callback = get_loss_callback()
    model, optimizer = get_mocks()

    sample_ids = [0, 1, 2, 3]
    data = torch.Tensor([0.5, 0.1, 0.2, 0.3])
    target = torch.Tensor([1, 2, 0, 1])
    output = torch.Tensor([1, 0, 1, 2])
    reduced_loss = torch.Tensor([1.0])

    loss_callback.on_batch_before_update(model, optimizer, 0, sample_ids, data, target, output, reduced_loss)
    assert loss_callback._sum_train_loss == 6.0
    test_add_per_sample_metadata_for_batch.assert_called_with(MetricType.LOSS, [0, 1, 2, 3], [0.0, 4.0, 1.0, 1.0])

    sample_ids_new = [4, 5, 6, 7]
    data_new = torch.Tensor([1.0, 0.1, 0.5, 0.3])
    target_new = torch.Tensor([2, 4, 0, 2])
    output_new = torch.Tensor([1, 3, 1, 2])
    reduced_loss_new = torch.Tensor([0.5])

    loss_callback.on_batch_before_update(
        model, optimizer, 0, sample_ids_new, data_new, target_new, output_new, reduced_loss_new
    )
    assert loss_callback._sum_train_loss == 9.0
    test_add_per_sample_metadata_for_batch.assert_called_with(MetricType.LOSS, [4, 5, 6, 7], [1.0, 1.0, 1.0, 0.0])


@patch.object(MetadataCollector, "add_per_trigger_metadata", return_value=None)
def test_on_train_end(test_add_per_trigger_metadata):
    model, optimizer = get_mocks()
    loss_callback = get_loss_callback()
    loss_callback._sum_train_loss = 10.0
    loss_callback.on_train_end(model, optimizer, 10, 1)
    assert loss_callback._average_train_loss == 1.0
    test_add_per_trigger_metadata.assert_called_once_with(MetricType.LOSS, 1.0)
