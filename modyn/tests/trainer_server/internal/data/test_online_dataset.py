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


def bytes_parser_function(data):
    return data


class MockSelectorStub:
    def __init__(self, channel) -> None:
        pass

    def get_sample_keys_and_weights(self, request):
        return SamplesResponse(training_samples_subset=["1", "2", "3"], training_samples_weights=[1.0, 1.0, 1.0])


class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request):  # pylint: disable=invalid-name
        for i in range(0, 10, 2):
            yield GetResponse(
                samples=[bytes(f"sample{i}", "utf-8"), bytes(f"sample{i+1}", "utf-8")],
                keys=[str(i), str(i + 1)],
                labels=[i, i + 1],
            )


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
            training_id=42,
        )._init_transforms()

    with pytest.raises(ValueError, match="Missing function bytes_parser_function from training invocation"):
        OnlineDataset(
            pipeline_id=1,
            trigger_id=1,
            dataset_id="MNIST",
            bytes_parser="bytes_parser_function=1",
            serialized_transforms=[],
            storage_address="localhost:1234",
            selector_address="localhost:1234",
            training_id=42,
        )._init_transforms()


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
        training_id=42,
    )
    assert online_dataset._pipeline_id == 1
    assert online_dataset._trigger_id == 1
    assert online_dataset._dataset_id == "MNIST"
    assert online_dataset._dataset_len == 0
    assert online_dataset._trainining_set_number == 0
    assert online_dataset._bytes_parser_function is None
    assert online_dataset._selectorstub is None
    assert online_dataset._storagestub is None


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
        training_id=42,
    )

    online_dataset._init_grpc()
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
        training_id=42,
    )
    online_dataset._init_grpc()
    assert online_dataset._get_data_from_storage([str(x) for x in range(10)]) == (
        [bytes(f"sample{x}", "utf-8") for x in range(10)],
        list(range(10)),
    )

    permuted_list = ["0", "9", "6", "5", "4", "3"]
    assert online_dataset._get_data_from_storage(permuted_list) == (
        [b"sample0", b"sample9", b"sample6", b"sample5", b"sample4", b"sample3"],
        [0, 9, 6, 5, 4, 3],
    )


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
        training_id=42,
    )
    online_dataset._bytes_parser_function = bytes_parser_function
    online_dataset._deserialize_torchvision_transforms()
    assert isinstance(online_dataset._transform.transforms, list)
    assert online_dataset._transform.transforms[0].__name__ == "bytes_parser_function"
    for transform1, transform2 in zip(online_dataset._transform.transforms[1:], transforms_list):
        assert transform1.__dict__ == transform2.__dict__


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([bytes(f"sample{x}", "utf-8") for x in range(10)], [1] * 10)
)
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[str(i) for i in range(10)])
def test_dataset_iter(test_get_keys, test_get_data, test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == [str(i) for i in range(10)]
    assert [x[1] for x in all_data] == [bytes(f"sample{x}", "utf-8") for x in range(10)]
    assert [x[2] for x in all_data] == [1] * 10


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([bytes(f"sample{x}", "utf-8") for x in range(10)], [1] * 10)
)
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[str(i) for i in range(10)])
def test_dataset_iter_with_parsing(
    test_get_data, test_get_keys, test_insecure_channel, test_grpc_connection_established
):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn x.decode('utf-8')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == [str(i) for i in range(10)]
    assert [x[1] for x in all_data] == [f"sample{i}" for i in range(10)]
    assert [x[2] for x in all_data] == [1] * 10


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([x.to_bytes(2, "big") for x in range(16)], [1] * 16)
)
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[str(i) for i in range(16)])
def test_dataloader_dataset(test_get_data, test_get_keys, test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert batch[0] == (str(4 * i), str(4 * i + 1), str(4 * i + 2), str(4 * i + 3))
        assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", return_value=([x.to_bytes(2, "big") for x in range(4)], [1] * 4))
@patch.object(OnlineDataset, "_get_keys_from_selector", return_value=[str(i) for i in range(4)])
def test_dataloader_dataset_multi_worker(
    test_get_data, test_get_keys, test_insecure_channel, test_grpc_connection_established
):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4, num_workers=4)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert batch[0] == ("0", "1", "2", "3")
        assert torch.equal(batch[1], torch.Tensor([0, 1, 2, 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=int))

@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_grpc(test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
    )

    assert online_dataset._selectorstub is None
    assert online_dataset._storagestub is None

    online_dataset._init_grpc()

    assert isinstance(online_dataset._selectorstub, MockSelectorStub)
    assert isinstance(online_dataset._storagestub, MockStorageStub)


@patch("modyn.trainer_server.internal.dataset.online_dataset.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_transforms(test_insecure_channel, test_grpc_connection_established):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
    )

    assert online_dataset._bytes_parser_function is None
    assert online_dataset._transform is None

    with patch.object(online_dataset, "_deserialize_torchvision_transforms") as tv_ds:
        online_dataset._init_transforms()
        assert online_dataset._bytes_parser_function is not None
        assert online_dataset._bytes_parser_function(b"\x01") == 1

        assert online_dataset._transform is not None

        tv_ds.assert_called_once()
