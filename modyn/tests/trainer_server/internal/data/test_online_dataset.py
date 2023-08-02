# pylint: disable=unused-argument, no-name-in-module
import platform
from unittest.mock import patch

import grpc
import pytest
import torch
from modyn.selector.internal.grpc.generated.selector_pb2 import SamplesResponse, UsesWeightsResponse
from modyn.storage.internal.grpc.generated.storage_pb2 import GetResponse
from modyn.trainer_server.internal.dataset.key_sources import SelectorKeySource
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
        return [SamplesResponse(training_samples_subset=[1, 2, 3], training_samples_weights=[1.0, 1.0, 1.0])]

    def uses_weights(self, request):
        return UsesWeightsResponse(uses_weights=False)


class WeightedMockSelectorStub(MockSelectorStub):
    def uses_weights(self, request):
        return UsesWeightsResponse(uses_weights=True)


class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request):  # pylint: disable=invalid-name
        for i in range(0, 10, 2):
            yield GetResponse(
                samples=[bytes(f"sample{i}", "utf-8"), bytes(f"sample{i+1}", "utf-8")],
                keys=[i, i + 1],
                labels=[i, i + 1],
            )


@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch.object(SelectorKeySource, "uses_weights", return_value=False)
def test_invalid_bytes_parser(test_weights, test_grpc_connection_established):
    with pytest.raises(AssertionError):
        OnlineDataset(
            pipeline_id=1,
            trigger_id=1,
            dataset_id="MNIST",
            bytes_parser="",
            serialized_transforms=[],
            storage_address="localhost:1234",
            selector_address="localhost:1234",
            training_id=42,
            tokenizer=None,
        )._init_transforms()

    with pytest.raises(ValueError):
        OnlineDataset(
            pipeline_id=1,
            trigger_id=1,
            dataset_id="MNIST",
            bytes_parser="bytes_parser_function=1",
            serialized_transforms=[],
            storage_address="localhost:1234",
            selector_address="localhost:1234",
            training_id=42,
            tokenizer="",
        )._init_transforms()


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init(test_insecure_channel, test_grpc_connection_established, test_grpc_connection_established_selector):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
    )
    assert online_dataset._pipeline_id == 1
    assert online_dataset._trigger_id == 1
    assert online_dataset._dataset_id == "MNIST"
    assert online_dataset._first_call
    assert online_dataset._bytes_parser_function is None
    assert online_dataset._storagestub is None


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_keys_and_weights_from_selector(
    test_insecure_channel, test_grpc_connection_established, test_grpc_connection_established_selector
):
    for return_weights in [True, False]:
        kwargs = {
            "pipeline_id": 1,
            "trigger_id": 1,
            "dataset_id": "MNIST",
            "bytes_parser": get_mock_bytes_parser(),
            "serialized_transforms": [],
            "storage_address": "localhost:1234",
            "selector_address": "localhost:1234",
            "training_id": 42,
            "tokenizer": None,
        }

        online_dataset = OnlineDataset(**kwargs)

        online_dataset._key_source._uses_weights = return_weights
        online_dataset._init_grpc()
        online_dataset._key_source.init_worker()
        keys, weights = online_dataset._key_source.get_keys_and_weights(0, 0)
        assert keys == [1, 2, 3]
        assert weights == [1.0, 1.0, 1.0] if return_weights else weights is None


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_data_from_storage(
    test_insecure_channel, test_grpc_connection_established, test_grpc_connection_established_selector
):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
    )
    online_dataset._init_grpc()
    assert online_dataset._get_data_from_storage(list(range(10))) == (
        [bytes(f"sample{x}", "utf-8") for x in range(10)],
        list(range(10)),
    )

    permuted_list = [0, 9, 6, 5, 4, 3]
    assert online_dataset._get_data_from_storage(permuted_list) == (
        [b"sample0", b"sample9", b"sample6", b"sample5", b"sample4", b"sample3"],
        [0, 9, 6, 5, 4, 3],
    )


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
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
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
    serialized_transforms,
    transforms_list,
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
        tokenizer=None,
    )
    online_dataset._bytes_parser_function = bytes_parser_function
    online_dataset._setup_composed_transform()
    assert isinstance(online_dataset._transform.transforms, list)
    assert online_dataset._transform.transforms[0].__name__ == "bytes_parser_function"
    for transform1, transform2 in zip(online_dataset._transform.transforms[1:], transforms_list):
        assert transform1.__dict__ == transform2.__dict__


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([bytes(f"sample{x}", "utf-8") for x in range(10)], [1] * 10)
)
@patch.object(SelectorKeySource, "get_keys_and_weights", return_value=(list(range(10)), None))
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=1)
def test_dataset_iter(
    test_get_num_data_partitions,
    test_get_keys,
    test_get_data,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == list(range(10))
    assert [x[1] for x in all_data] == [bytes(f"sample{x}", "utf-8") for x in range(10)]
    assert [x[2] for x in all_data] == [1] * 10


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([bytes(f"sample{x}", "utf-8") for x in range(10)], [1] * 10)
)
@patch.object(SelectorKeySource, "get_keys_and_weights", return_value=(list(range(10)), None))
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=1)
def test_dataset_iter_with_parsing(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
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
        tokenizer=None,
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == list(range(10))
    assert [x[1] for x in all_data] == [f"sample{i}" for i in range(10)]
    assert [x[2] for x in all_data] == [1] * 10


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([x.to_bytes(2, "big") for x in range(16)], [1] * 16)
)
@patch.object(SelectorKeySource, "get_keys_and_weights", return_value=(list(range(16)), None))
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=1)
def test_dataloader_dataset(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
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
        tokenizer=None,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert batch[0].tolist() == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", WeightedMockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([x.to_bytes(2, "big") for x in range(16)], [1] * 16)
)
@patch.object(SelectorKeySource, "get_keys_and_weights", return_value=(list(range(16)), [2.0] * 16))
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=1)
def test_dataloader_dataset_weighted(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
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
        tokenizer=None,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 4
        assert batch[0].tolist() == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
        assert torch.equal(batch[3], 2 * torch.ones(4, dtype=torch.float64))


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", return_value=([x.to_bytes(2, "big") for x in range(4)], [1] * 4))
@patch.object(SelectorKeySource, "get_keys_and_weights", return_value=(list(range(4)), None))
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=1)
def test_dataloader_dataset_multi_worker(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
):
    if platform.system() == "Darwin":
        # On macOS, spawn is the default, which loses the mocks
        # Hence the test does not work on macOS, only on Linux.
        return

    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4, num_workers=4)
    for batch in dataloader:
        assert len(batch) == 3
        assert torch.equal(batch[0], torch.Tensor([0, 1, 2, 3]))
        assert torch.equal(batch[1], torch.Tensor([0, 1, 2, 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=int))


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_grpc(test_insecure_channel, test_grpc_connection_established, test_grpc_connection_established_selector):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
    )

    assert online_dataset._storagestub is None

    online_dataset._init_grpc()
    online_dataset._key_source.init_worker()

    assert isinstance(online_dataset._key_source._selectorstub, MockSelectorStub)
    assert isinstance(online_dataset._storagestub, MockStorageStub)


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_init_transforms(
    test_insecure_channel, test_grpc_connection_established, test_grpc_connection_established_selector
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
        tokenizer=None,
    )

    assert online_dataset._bytes_parser_function is None
    assert online_dataset._transform is None

    with patch.object(online_dataset, "_setup_composed_transform") as tv_ds:
        online_dataset._init_transforms()
        assert online_dataset._bytes_parser_function is not None
        assert online_dataset._bytes_parser_function(b"\x01") == 1

        assert online_dataset._transform is not None

        tv_ds.assert_called_once()


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset,
    "_get_data_from_storage",
    side_effect=[
        ([x.to_bytes(2, "big") for x in range(16)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(16, 32)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(32, 48)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(48, 64)], [1] * 16),
    ],
)
@patch.object(
    SelectorKeySource,
    "get_keys_and_weights",
    side_effect=[
        ([str(i) for i in range(16)], None),
        ([str(i) for i in range(16, 32)], None),
        ([str(i) for i in range(32, 48)], None),
        ([str(i) for i in range(48, 64)], None),
    ],
)
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=4)
def test_iter_multi_partition(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
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
        tokenizer=None,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)

    idx = 0
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert batch[0] == (str(4 * idx), str(4 * idx + 1), str(4 * idx + 2), str(4 * idx + 3))
        assert torch.equal(batch[1], torch.Tensor([4 * idx, 4 * idx + 1, 4 * idx + 2, 4 * idx + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
    assert idx == 15


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", WeightedMockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset,
    "_get_data_from_storage",
    side_effect=[
        ([x.to_bytes(2, "big") for x in range(16)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(16, 32)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(32, 48)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(48, 64)], [1] * 16),
    ],
)
@patch.object(
    SelectorKeySource,
    "get_keys_and_weights",
    side_effect=[
        ([str(i) for i in range(16)], [0.9] * 16),
        ([str(i) for i in range(16, 32)], [0.9] * 16),
        ([str(i) for i in range(32, 48)], [0.9] * 16),
        ([str(i) for i in range(48, 64)], [0.9] * 16),
    ],
)
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=4)
def test_iter_multi_partition_weighted(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
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
        tokenizer=None,
    )

    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)

    idx = 0
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 4
        assert batch[0] == (str(4 * idx), str(4 * idx + 1), str(4 * idx + 2), str(4 * idx + 3))
        assert torch.equal(batch[1], torch.Tensor([4 * idx, 4 * idx + 1, 4 * idx + 2, 4 * idx + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
        assert torch.equal(batch[3], 0.9 * torch.ones(4, dtype=torch.float64))
    assert idx == 15


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset,
    "_get_data_from_storage",
    side_effect=[
        ([x.to_bytes(2, "big") for x in range(16)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(16, 32)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(32, 48)], [1] * 16),
        ([x.to_bytes(2, "big") for x in range(48, 64)], [1] * 16),
    ],
)
@patch.object(
    SelectorKeySource,
    "get_keys_and_weights",
    side_effect=[
        ([str(i) for i in range(16)], None),
        ([str(i) for i in range(16, 32)], None),
        ([str(i) for i in range(32, 48)], None),
        ([str(i) for i in range(48, 64)], None),
    ],
)
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=4)
def test_iter_multi_partition_cross(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
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
        tokenizer=None,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=6)

    idx = 0
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 3
        if idx < 10:
            assert batch[0] == (
                str(6 * idx),
                str(6 * idx + 1),
                str(6 * idx + 2),
                str(6 * idx + 3),
                str(6 * idx + 4),
                str(6 * idx + 5),
            )
            assert torch.equal(
                batch[1], torch.Tensor([6 * idx, 6 * idx + 1, 6 * idx + 2, 6 * idx + 3, 6 * idx + 4, 6 * idx + 5])
            )
            assert torch.equal(batch[2], torch.ones(6, dtype=torch.float64))
        else:
            assert batch[0] == ("60", "61", "62", "63")
            assert torch.equal(batch[1], torch.Tensor([60, 61, 62, 63]))
            assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
    assert idx == 10


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset,
    "_get_data_from_storage",
    side_effect=[
        ([x.to_bytes(2, "big") for x in range(4)], [1] * 4),
        ([x.to_bytes(2, "big") for x in range(4)], [1] * 4),
    ],
)
@patch.object(
    SelectorKeySource,
    "get_keys_and_weights",
    side_effect=[(list(range(4)), [1.0] * 4), (list(range(4)), [1.0] * 4)],
)
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=2)
def test_iter_multi_partition_multi_workers(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
):
    if platform.system() == "Darwin":
        # On macOS, spawn is the default, which loses the mocks
        # Hence the test does not work on macOS, only on Linux.
        return

    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4, num_workers=4)
    idx = 0
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert torch.equal(batch[0], torch.Tensor([0, 1, 2, 3]))
        assert torch.equal(batch[1], torch.Tensor([0, 1, 2, 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=int))
    assert idx == 7


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(
    OnlineDataset, "_get_data_from_storage", return_value=([x.to_bytes(2, "big") for x in range(100)], [1] * 100)
)
@patch.object(SelectorKeySource, "get_keys_and_weights", return_value=(list(range(100)), None))
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=1)
def test_multi_epoch_dataloader_dataset(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selecotr,
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
        tokenizer=None,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for _ in range(5):
        for i, batch in enumerate(dataloader):
            assert len(batch) == 3
            assert batch[0].tolist() == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
            assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
            assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
