# pylint: disable=unused-argument
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset

from modyn.trainer_server.internal.dataset.utils import prepare_dataloaders


def test_prepare_dataloaders():

    train_dataloader, _ = prepare_dataloaders(
        1,
        "MNIST",
        4,
        128,
        [],
        "new"
    )

    assert train_dataloader.num_workers == 4
    assert train_dataloader.batch_size == 128
    assert isinstance(train_dataloader.dataset, OnlineDataset)
