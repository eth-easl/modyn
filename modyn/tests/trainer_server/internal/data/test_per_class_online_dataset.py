# pylint: disable=unused-argument, no-name-in-module
# ruff: noqa: N802  # grpc functions are not snake case

from unittest.mock import patch

import grpc
import pytest
import torch

from modyn.selector.internal.grpc.generated.selector_pb2 import SamplesResponse, UsesWeightsResponse
from modyn.storage.internal.grpc.generated.storage_pb2 import GetResponse
from modyn.trainer_server.internal.dataset.key_sources import SelectorKeySource
from modyn.trainer_server.internal.dataset.per_class_online_dataset import PerClassOnlineDataset


class MockSelectorStub:
    def __init__(self, channel) -> None:
        pass

    def get_sample_keys_and_weights(self, request):
        return [SamplesResponse(training_samples_subset=[1, 2, 3], training_samples_weights=[1.0, 1.0, 1.0])]

    def uses_weights(self, request):
        return UsesWeightsResponse(uses_weights=False)


class MockStorageStub:
    def __init__(self, channel) -> None:
        pass

    def Get(self, request):  # pylint: disable=invalid-name
        for i in range(0, 10, 2):
            yield GetResponse(
                samples=[bytes(f"sample{i}", "utf-8"), bytes(f"sample{i+1}", "utf-8")],
                keys=[i, i + 1],
                labels=[i, i + 1],
                target=[bytes(f"sample{i}", "utf-8"), bytes(f"sample{i+1}", "utf-8")],
            )


@pytest.mark.parametrize("parallel_prefetch_requests", [1, 2, 5, 7, 8, 9, 10, 100, 999999])
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
    PerClassOnlineDataset,
    "_get_data_from_storage",
    return_value=[
        (
            list(range(16)),
            [x.to_bytes(2, "big") for x in range(16)],
            [0, 1, 2, 3, 0, 0, 0, 1] * 2,
            [x.to_bytes(2, "big") for x in range(16)],
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
    online_dataset = PerClassOnlineDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser="def bytes_parser_function(x):\n\treturn int.from_bytes(x, 'big')",
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        initial_filtered_label=0,
        num_prefetched_partitions=prefetched_partitions,
        parallel_prefetch_requests=parallel_prefetch_requests,
        tokenizer=None,
        shuffle=False,
    )
    dataloader = torch.utils.data.DataLoader(online_dataset, batch_size=4)

    # samples with 0 as label = 0, 4, 5, 6, 8, 12, 13, 14
    samples = []
    for batch in dataloader:
        samples += batch[0].tolist()
    assert samples == [0, 4, 5, 6, 8, 12, 13, 14]

    # samples with 1 as label: 1, 7, 9, 15
    dataloader.dataset.filtered_label = 1
    samples = []
    for batch in dataloader:
        samples += batch[0].tolist()
    assert samples == [1, 7, 9, 15]

    # samples with 2 as label: 2, 10
    dataloader.dataset.filtered_label = 2
    samples = []
    for batch in dataloader:
        samples += batch[0].tolist()
    assert samples == [2, 10]

    # samples with 3 as label: 3, 11
    dataloader.dataset.filtered_label = 3
    samples = []
    for batch in dataloader:
        samples += batch[0].tolist()
    assert samples == [3, 11]
