# pylint: disable=unused-argument
from unittest.mock import patch

import grpc
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.dataset.data_utils import prepare_dataloaders
from modyn.trainer_server.internal.dataset.key_sources import SelectorKeySource
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


def noop_constructor_mock(self, channel: grpc.Channel) -> None:
    pass


@patch.object(StorageStub, "__init__", noop_constructor_mock)
@patch.object(SelectorStub, "__init__", noop_constructor_mock)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(SelectorKeySource, "uses_weights", return_value=False)
def test_prepare_dataloaders(
    test_weights, test_insecure_channel, test_grpc_connection_established, test_grpc_connection_established_selector
):
    train_dataloader, _ = prepare_dataloaders(1, 1, "MNIST", 4, 128, get_mock_bytes_parser(), [], "", "", 42, "")

    assert train_dataloader.num_workers == 4
    assert train_dataloader.batch_size == 128
    assert isinstance(train_dataloader.dataset, OnlineDataset)
