# pylint: disable=unused-argument
from modyn.trainer_server.internal.dataset.data_utils import prepare_dataloaders
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


def test_prepare_dataloaders():
    train_dataloader, _ = prepare_dataloaders(1, "MNIST", 4, 128, get_mock_bytes_parser(), [], "new")

    assert train_dataloader.num_workers == 4
    assert train_dataloader.batch_size == 128
    assert isinstance(train_dataloader.dataset, OnlineDataset)
