# pylint: disable=unused-argument, no-name-in-module, no-value-for-parameter
import pathlib
from typing import Generator, Optional
from unittest.mock import patch

from modyn.supervisor.internal.triggers.trigger_datasets import OnlineTriggerDataset
from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset

NUM_SAMPLES = 10


def get_mock_bytes_parser():
    return "def bytes_parser_function(x):\n\treturn x"


def bytes_parser_function(data):
    return data


def noop_constructor_mock(
    self,
    pipeline_id: int,
    trigger_id: int,
    dataset_id: str,
    bytes_parser: str,
    serialized_transforms: list[str],
    storage_address: str,
    selector_address: str,
    training_id: int,
    num_prefetched_partitions: int,
    parallel_prefetch_requests: int,
    tokenizer: Optional[str],
    log_path: Optional[pathlib.Path],
) -> None:
    pass


def mock_data_generator(self) -> Generator:
    yield from list(range(NUM_SAMPLES))


def test_init():
    online_trigger_dataset = OnlineTriggerDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
        num_prefetched_partitions=1,
        parallel_prefetch_requests=1,
        sample_prob=0.5,
        shuffle=False,
    )
    assert online_trigger_dataset._pipeline_id == 1
    assert online_trigger_dataset._trigger_id == 1
    assert online_trigger_dataset._dataset_id == "MNIST"
    assert online_trigger_dataset._first_call
    assert online_trigger_dataset._bytes_parser_function is None
    assert online_trigger_dataset._storagestub is None
    assert online_trigger_dataset._sample_prob == 0.5


@patch.object(OnlineDataset, "__iter__", mock_data_generator)
def test_dataset_iter():
    online_trigger_dataset = OnlineTriggerDataset(
        pipeline_id=1,
        trigger_id=1,
        dataset_id="MNIST",
        bytes_parser=get_mock_bytes_parser(),
        serialized_transforms=[],
        storage_address="localhost:1234",
        selector_address="localhost:1234",
        training_id=42,
        tokenizer=None,
        num_prefetched_partitions=1,
        parallel_prefetch_requests=1,
        sample_prob=0.5,
        shuffle=False,
    )

    all_trigger_data = list(online_trigger_dataset)
    assert len(all_trigger_data) < NUM_SAMPLES
