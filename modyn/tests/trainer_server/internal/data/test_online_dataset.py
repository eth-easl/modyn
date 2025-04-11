# pylint: disable=unused-argument, no-name-in-module, too-many-locals

import platform
from unittest.mock import MagicMock, patch

import grpc
import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from modyn.selector.internal.grpc.generated.selector_pb2 import SamplesResponse, UsesWeightsResponse
from modyn.storage.internal.grpc.generated.storage_pb2 import GetResponse
from modyn.trainer_server.internal.dataset.key_sources import AbstractKeySource, SelectorKeySource
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset


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

    def Get(self, request):  # pylint: disable=invalid-name  # noqa: N802
        for i in range(0, 10, 2):
            yield GetResponse(
                samples=[bytes(f"sample{i}", "utf-8"), bytes(f"sample{i+1}", "utf-8")],
                keys=[i, i + 1],
                labels=[i, i + 1],
                target=[bytes(f"sample{i}", "utf-8"), bytes(f"sample{i+1}", "utf-8")],
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
            log_path=None,
            shuffle=False,
            num_prefetched_partitions=1,
            parallel_prefetch_requests=1,
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
            log_path=None,
            shuffle=False,
            num_prefetched_partitions=1,
            parallel_prefetch_requests=1,
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
        log_path=None,
        shuffle=False,
        num_prefetched_partitions=1,
        parallel_prefetch_requests=1,
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
            "log_path": None,
            "shuffle": False,
            "num_prefetched_partitions": 1,
            "parallel_prefetch_requests": 1,
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
        log_path=None,
        shuffle=False,
        num_prefetched_partitions=0,
        parallel_prefetch_requests=1,
    )
    online_dataset._init_grpc()
    keys = []
    data = []
    labels = []
    targets = []
    for key_list, data_list, label_list, gen_targets_list, _ in online_dataset._get_data_from_storage(list(range(10))):
        keys.extend(key_list)
        data.extend(data_list)
        labels.extend(label_list)
        targets.extend(gen_targets_list)
    assert (keys, data, labels, targets) == (
        list(range(10)),
        [bytes(f"sample{x}", "utf-8") for x in range(10)],
        list(range(10)),
        [bytes(f"sample{x}", "utf-8") for x in range(10)],
    )

    result_keys = []
    result_samples = []
    result_labels = []
    result_targets = []
    permuted_list = [0, 9, 6, 5, 4, 3]
    for rkey, rsam, rlbl, rtrgt, _ in online_dataset._get_data_from_storage(permuted_list):
        result_keys.extend(rkey)
        result_samples.extend(rsam)

        result_labels.extend(rlbl)
        result_targets.extend(rtrgt)

    assert set(result_keys) == set(keys)
    assert set(result_samples) == set(data)
    assert set(result_labels) == set(labels)
    assert set(result_targets) == set(targets)


class MockRpcError(grpc.RpcError):
    def code(self):
        return grpc.StatusCode.UNAVAILABLE

    def details(self):
        return "Mocked gRPC error for testing retry logic."


@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub")
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_get_data_from_storage_with_retry(
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
    mock_storage_stub,
):
    # Arrange
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
        log_path=None,
        shuffle=False,
        num_prefetched_partitions=0,
        parallel_prefetch_requests=1,
    )
    online_dataset._init_grpc = MagicMock()  # cannot patch this with annotations due to tenacity
    mock_storage_stub.Get.side_effect = [MockRpcError(), MockRpcError(), MagicMock()]
    online_dataset._storagestub = mock_storage_stub

    try:
        for _ in online_dataset._get_data_from_storage(list(range(10))):
            pass
    except Exception as e:
        assert isinstance(e, RuntimeError), "Expected a RuntimeError after max retries."

    assert mock_storage_stub.Get.call_count == 3, "StorageStub.Get should have been retried twice before succeeding."


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
        log_path=None,
        shuffle=False,
        num_prefetched_partitions=1,
        parallel_prefetch_requests=1,
    )
    online_dataset._bytes_parser_function = bytes_parser_function
    online_dataset._setup_composed_transform()
    assert isinstance(online_dataset._transform.transforms, list)
    assert online_dataset._transform.transforms[0].__name__ == "bytes_parser_function"
    for transform1, transform2 in zip(online_dataset._transform.transforms[1:], transforms_list):
        assert transform1.__dict__ == transform2.__dict__


@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
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
    return_value=[
        (
            list(range(10)),
            [bytes(f"sample{x}", "utf-8") for x in range(10)],
            [1] * 10,
            [bytes(f"sample{x}", "utf-8") for x in range(10)],
            0,
        )
    ],
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
    prefetched_partitions,
    parallel_prefetch_requests,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=False,
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == list(range(10))
    assert [x[1] for x in all_data] == [bytes(f"sample{x}", "utf-8") for x in range(10)]
    assert [x[2] for x in all_data] == [1] * 10


@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
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
    return_value=[
        (
            list(range(10)),
            [bytes(f"sample{x}", "utf-8") for x in range(10)],
            [1] * 10,
            [bytes(f"sample{x}", "utf-8") for x in range(10)],
            0,
        )
    ],
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
    prefetched_partitions,
    parallel_prefetch_requests,
):
    online_dataset = OnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x: memoryview):\n\treturn x.tobytes().decode('utf-8')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=False,
    )
    dataset_iter = iter(online_dataset)
    all_data = list(dataset_iter)
    assert [x[0] for x in all_data] == list(range(10))
    assert [x[1] for x in all_data] == [f"sample{i}" for i in range(10)]
    assert [x[2] for x in all_data] == [1] * 10


@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
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
    return_value=[
        (
            list(range(16)),
            [x.to_bytes(2, "big") for x in range(16)],
            [1] * 16,
            [bytes(f"sample{x}", "utf-8") for x in range(10)],
            0,
        )
    ],
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
    prefetched_partitions,
    parallel_prefetch_requests,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert batch[0].tolist() == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
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
    return_value=[
        (
            list(range(16)),
            [x.to_bytes(2, "big") for x in range(16)],
            [1] * 16,
            [x.to_bytes(2, "big") for x in range(2)],
            0,
        )
    ],
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
    prefetched_partitions,
    parallel_prefetch_requests,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for i, batch in enumerate(dataloader):
        assert len(batch) == 4
        assert batch[0].tolist() == [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
        assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
        assert torch.equal(batch[3], 2 * torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
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
    return_value=[
        (list(range(4)), [x.to_bytes(2, "big") for x in range(4)], [1] * 4, [x.to_bytes(2, "big") for x in range(2)], 0)
    ],
)
@patch.object(SelectorKeySource, "get_keys_and_weights", return_value=(list(range(4)), None))
@patch.object(SelectorKeySource, "get_num_data_partitions", return_value=1)
def test_dataloader_dataset_multi_worker(
    test_get_num_data_partitions,
    test_get_data,
    test_get_keys,
    test_insecure_channel,
    test_grpc_connection_established,
    test_grpc_connection_established_selector,
    prefetched_partitions,
    num_workers,
    parallel_prefetch_requests,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4, num_workers=num_workers)
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
        num_prefetched_partitions=1,
        parallel_prefetch_requests=1,
        tokenizer=None,
        log_path=None,
        shuffle=False,
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
        num_prefetched_partitions=1,
        parallel_prefetch_requests=1,
        tokenizer=None,
        log_path=None,
        shuffle=False,
    )

    assert online_dataset._bytes_parser_function is None
    assert online_dataset._transform is None

    with patch.object(online_dataset, "_setup_composed_transform") as tv_ds:
        online_dataset._init_transforms()
        assert online_dataset._bytes_parser_function is not None
        assert online_dataset._bytes_parser_function(b"\x01") == 1

        assert online_dataset._transform is not None

        tv_ds.assert_called_once()


def iter_multi_partition_data_side_effect(keys, worker_id=None):
    yield (list(keys), [x.to_bytes(2, "big") for x in keys], [1] * len(keys), [x.to_bytes(2, "big") for x in keys], 0)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", side_effect=iter_multi_partition_data_side_effect)
@patch.object(
    SelectorKeySource,
    "get_keys_and_weights",
    side_effect=[
        (list(range(16)), None),
        (list(range(16, 32)), None),
        (list(range(32, 48)), None),
        (list(range(48, 64)), None),
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
    prefetched_partitions,
    parallel_prefetch_requests,
    shuffle,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=shuffle,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    idx = 0
    all_samples = []
    all_data = []
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 3
        all_samples.extend(batch[0].tolist())
        all_data.extend(batch[1].tolist())
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
    expected_samples = list(range(64))
    expected_data = expected_samples
    assert set(all_samples) == set(expected_samples)
    assert set(all_data) == set(expected_data)
    assert idx == 15


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", WeightedMockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", side_effect=iter_multi_partition_data_side_effect)
@patch.object(
    SelectorKeySource,
    "get_keys_and_weights",
    side_effect=[
        (list(range(16)), [0.9] * 16),
        (list(range(16, 32)), [0.9] * 16),
        (list(range(32, 48)), [0.9] * 16),
        (list(range(48, 64)), [0.9] * 16),
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
    prefetched_partitions,
    parallel_prefetch_requests,
    shuffle,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=shuffle,
    )

    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)

    idx = 0
    all_samples = []
    all_data = []
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 4
        all_samples.extend(batch[0].tolist())
        all_data.extend(batch[1].tolist())
        assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
        assert torch.equal(batch[3], 0.9 * torch.ones(4, dtype=torch.float64))
    expected_samples = list(range(64))
    expected_data = expected_samples
    assert set(all_samples) == set(expected_samples)
    assert set(all_data) == set(expected_data)
    assert idx == 15


@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
@patch("modyn.trainer_server.internal.dataset.key_sources.selector_key_source.SelectorStub", MockSelectorStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", MockStorageStub)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch.object(grpc, "insecure_channel", return_value=None)
@patch.object(OnlineDataset, "_get_data_from_storage", side_effect=iter_multi_partition_data_side_effect)
@patch.object(
    SelectorKeySource,
    "get_keys_and_weights",
    side_effect=[
        (list(range(16)), None),
        (list(range(16, 32)), None),
        (list(range(32, 48)), None),
        (list(range(48, 64)), None),
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
    prefetched_partitions,
    parallel_prefetch_requests,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=False,  # Since we test ordering here, shuffling does not make sense.
    )
    # Note batch size 6 instead of 4 here
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=6)

    idx = 0
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 3
        if idx < 10:
            assert torch.equal(
                batch[0], torch.Tensor([6 * idx, 6 * idx + 1, 6 * idx + 2, 6 * idx + 3, 6 * idx + 4, 6 * idx + 5])
            )
            assert torch.equal(
                batch[1], torch.Tensor([6 * idx, 6 * idx + 1, 6 * idx + 2, 6 * idx + 3, 6 * idx + 4, 6 * idx + 5])
            )
            assert torch.equal(batch[2], torch.ones(6, dtype=torch.float64))
        else:
            assert torch.equal(batch[0], torch.Tensor([60, 61, 62, 63]))
            assert torch.equal(batch[1], torch.Tensor([60, 61, 62, 63]))
            assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))
    assert idx == 10


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("num_workers", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
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
    side_effect=iter_multi_partition_data_side_effect,
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
    prefetched_partitions,
    num_workers,
    parallel_prefetch_requests,
    shuffle,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=shuffle,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4, num_workers=num_workers)
    idx = 0
    for idx, batch in enumerate(dataloader):
        assert len(batch) == 3
        assert torch.equal(batch[2], torch.ones(4, dtype=int))
        if shuffle:
            assert set(batch[0].tolist()) == set([0, 1, 2, 3])
            assert set(batch[1].tolist()) == set([0, 1, 2, 3])
        else:
            assert torch.equal(batch[0], torch.Tensor([0, 1, 2, 3]))
            assert torch.equal(batch[1], torch.Tensor([0, 1, 2, 3]))

    if num_workers % 2 == 0:
        # only test this for even number of workers to avoid fractions
        # each worker gets 8 items from get_keys_and_weights; batch size 4; minus one for zero indexing
        assert idx == ((max(num_workers, 1) * 8) / 4) - 1


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("parallel_prefetch_requests", [1, 5, 999999])
@pytest.mark.parametrize("prefetched_partitions", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 999999])
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
    return_value=iter(
        [
            (
                list(range(100)),
                [x.to_bytes(2, "big") for x in range(100)],
                [1] * 100,
                [x.to_bytes(2, "big") for x in range(100)],
                0,
            )
        ]
    ),
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
    prefetched_partitions,
    parallel_prefetch_requests,
    shuffle,
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
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        log_path=None,
        shuffle=shuffle,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)
    for _ in range(5):
        total_ids = []
        total_expected_ids = []
        total_data = []
        for i, batch in enumerate(dataloader):
            assert len(batch) == 3
            total_ids.extend(batch[0].tolist())
            expected_ids = [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
            total_expected_ids.extend(expected_ids)
            total_data.extend(batch[1])
            if not shuffle:
                assert batch[0].tolist() == expected_ids
                assert torch.equal(batch[1], torch.Tensor([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
                assert torch.equal(batch[2], torch.ones(4, dtype=torch.float64))

        if shuffle:
            assert set(total_ids) == set(total_expected_ids)
            for id in total_ids:
                for x in torch.cat(total_data):
                    assert id == int(x.item())


# Helper functions with proper formatting.
def get_main_bytes_parser() -> str:
    return """\
def bytes_parser_function(data: bytes):
    return bytes(data) + b"_MAIN"
"""


def get_target_bytes_parser() -> str:
    return """\
def bytes_parser_function(data: bytes):
    return bytes(data) + b"_TARGET"
"""


def get_main_serialized_transforms() -> list[str]:
    return ["transforms.Lambda(lambda x: x + b'_MAIN_TF')"]


def get_target_serialized_transforms() -> list[str]:
    return ["transforms.Lambda(lambda x: x + b'_TARGET_TF')"]


# A minimal KeySource for testing target parsing/transform logic.
class MockPartitionKeySource(AbstractKeySource):
    def __init__(self, pipeline_id=999, trigger_id=999):
        super().__init__(pipeline_id, trigger_id)

    def init_worker(self) -> None:
        pass

    def uses_weights(self) -> bool:
        return False

    def get_num_data_partitions(self) -> int:
        return 1

    def get_keys_and_weights(self, worker_id: int, partition_id: int):
        return [10, 11, 12], None

    def end_of_trigger_cleaning(self) -> None:
        pass


# Using decorator-style patching to simulate a fake gRPC connection.
@patch("modyn.trainer_server.internal.dataset.online_dataset.StorageStub", new=MockStorageStub)
@patch("modyn.trainer_server.internal.dataset.online_dataset.grpc_connection_established", return_value=True)
@patch(
    "modyn.trainer_server.internal.dataset.key_sources.selector_key_source.grpc_connection_established",
    return_value=True,
)
@patch.object(grpc, "insecure_channel", return_value=None)
def test_online_dataset_different_target_parsers_and_transforms(
    mock_insecure_channel, mock_selector_grpc_conn, mock_online_grpc_conn
):
    """
    Tests the scenario where:
      - The main bytes parser and transforms differ from those used for the target.
      - include_labels is False so that the dataset uses the target branch.

    We patch StorageStub, grpc_connection_established (in both modules), and grpc.insecure_channel.

    Note: We set collate_fn=lambda x: x[0] so that each batch (batch_size=1) directly returns the tuple.
    """
    dataset = OnlineDataset(
        pipeline_id=999,
        trigger_id=999,
        dataset_id="dummy_dataset",
        bytes_parser=get_main_bytes_parser(),
        bytes_parser_target=get_target_bytes_parser(),
        serialized_transforms=get_main_serialized_transforms(),
        serialized_transforms_target=get_target_serialized_transforms(),
        storage_address="mocked-storage:9999",
        selector_address="mocked-selector:9999",
        training_id=42,
        tokenizer=None,
        log_path=None,
        shuffle=False,
        num_prefetched_partitions=0,
        parallel_prefetch_requests=1,
        include_labels=False,  # Use target branch.
    )
    dataset.change_key_source(MockPartitionKeySource())

    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])
    data_items = list(loader)

    assert len(data_items) == 10

    for i, item in enumerate(data_items):
        assert isinstance(item, (tuple | list)), f"Expected item to be a tuple or list, got {type(item)}"
        assert len(item) == 3, f"Expected tuple of length 3, got {item}"
        key, sample, target = item
        expected_key = i
        assert key == expected_key, f"Expected key {expected_key}, got {key}"

        assert isinstance(sample, bytes), f"Expected sample to be bytes, got {type(sample)}"
        assert sample.endswith(b"_MAIN_MAIN_TF"), f"Got sample={sample}"

        assert isinstance(target, bytes), f"Expected target to be bytes, got {type(target)}"
        expected_suffix = b"_TARGET_TARGET_TF"
        assert target.endswith(expected_suffix), f"Got target={target}"
