# pylint: disable=unused-argument
import pytest
from modyn.gpu_node.data.online_dataset import OnlineDataset

from modyn.gpu_node.data.utils import prepare_dataloaders


def test_prepare_dataloaders_dataset_missing():

    with pytest.raises(AssertionError):
        prepare_dataloaders(1, "MissingDataset", 4, 128)


def test_prepare_dataloaders():

    train_dataloader, _ = prepare_dataloaders(
        1,
        "OnlineDataset",
        4,
        128
    )

    assert train_dataloader.num_workers == 4
    assert train_dataloader.batch_size == 128
    assert type(train_dataloader.dataset) == OnlineDataset
