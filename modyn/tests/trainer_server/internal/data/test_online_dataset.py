# pylint: disable=unused-argument, no-name-in-module
from unittest.mock import patch

import grpc
import pytest
import torch
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import SamplesResponse
from modyn.storage.internal.grpc.generated.storage_pb2 import GetResponse
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset
from torchvision import transforms


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


class MockSelectorStub:
    def __init__(self, channel) -> None:
        pass

    def get_sample_keys_and_weights(self, request):
        return SamplesResponse(training_samples_subset=["1", "2", "3"], training_samples_weights=[1.0, 1.0, 1.0])


class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request):  # pylint: disable=invalid-name
        return GetResponse(samples=[b"sample0", b"sample1"], keys=["1", "2"], labels=[0, 1])


def test_invalid_bytes_parser():
    with pytest.raises(ValueError, match="Missing function bytes_parser_function from training invocation"):
        OnlineDataset(
            pipeline_id=1,
            trigger_id=1,
            dataset_id="MNIST",
            bytes_parser="",
            serialized_transforms=[],
            storage_address="localhost:1234",
            selector_address="localhost:1234",
        )

    with pytest.raises(ValueError, match="Missing function bytes_parser_function from training invocation"):
        OnlineDataset(
            pipeline_id=1,
            trigger_id=1,
            dataset_id="MNIST",
            bytes_parser="bytes_parser_function=1",
            serialized_transforms=[],
            storage_address="localhost:1234",
            selector_address="localhost:1234",
        )


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init(test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
    )
    assert online_dataset._pipeline_id == 1
    assert online_dataset._trigger_id == 1
    assert online_dataset._dataset_id == "MNIST"
    assert online_dataset._dataset_len == 0
    assert online_dataset._trainining_set_number == 0
    assert online_dataset._bytes_parser_function
    assert online_dataset._transform.transforms[0].__name__ == "bytes_parser_function"
    assert isinstance(online_dataset._selectorstub, MockSelectorStub)
    assert isinstance(online_dataset._storagestub, MockStorageStub)


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_keys_from_selector(test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
    )
    assert online_dataset._get_keys_from_selector(0) == ["1", "2", "3"]


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_data_from_storage(test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
    )
    assert online_dataset._get_data_from_storage([]) == ([b"sample0", b"sample1"], [0, 1])


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@pytest.mark.parametrize(
    "serialized_transforms,transforms_list",
    [
        pytest.param(
            [
                "transforms.RandomResizedCrop(224)",
                "transforms.RandomHorizontalFlip()",
                "transforms.ToTensor()",
                "transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])",
            ],
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
        )
    ],
)
def test_deserialize_torchvision_transforms(
    test_insecure_channel, test_grpc_connection_established, serialized_transforms, transforms_list
):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=serialized_transforms,
        storage_address="localhost:1234",
        selector_address="localhost:1234",
    )
    online_dataset._deserialize_torchvision_transforms()
    assert isinstance(online_dataset._transform.transforms, list)
    assert online_dataset._transform.transforms[0].__name__ == "bytes_parser_function"
    for transform1, transform2 in zip(online_dataset._transform.transforms[1:], transforms_list):
        assert transform1.__dict__ == transform2.__dict__


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", return_value=(list(range(10)), [1] * 10))
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[])
def test_dataset_iter(test_get_keys, test_get_data, test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == list(range(10))
    assert [x[1] for x in all_data] == [1] * 10


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", return_value=(list(range(10)), [1] * 10))
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[])
def test_dataset_iter_with_parsing(
    test_get_data, test_get_keys, test_insecure_channel, test_grpc_connection_established
):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn 2*x",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == list(range(0, 20, 2))
    assert [x[1] for x in all_data] == [1] * 10


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", return_value=([0] * 16, [1] * 16))
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[])
def test_dataloader_dataset(test_get_data, test_get_keys, test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for batch in dataloader:
        assert len(batch) == 2
        assert torch.equal(batch[0], torch.zeros(4, dtype=int))
        assert torch.equal(batch[1], torch.ones(4, dtype=int))
