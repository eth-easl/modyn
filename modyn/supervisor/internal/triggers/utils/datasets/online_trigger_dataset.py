import logging
import random
from collections.abc import Generator

from torch.utils.data import IterableDataset

from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset

logger = logging.getLogger(__name__)


class OnlineTriggerDataset(OnlineDataset, IterableDataset):
    def __init__(
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
        shuffle: bool,
        tokenizer: str | None = None,
        sample_prob: float | None = None,
    ):
        # Updated to match OnlineDataset's __init__ signature:
        # (bytes_parser_target, serialized_transforms_target, log_path, include_labels)
        OnlineDataset.__init__(
            self,
            pipeline_id,
            trigger_id,
            dataset_id,
            bytes_parser,
            None,  # bytes_parser_target
            serialized_transforms,
            None,  # serialized_transforms_target
            storage_address,
            selector_address,
            training_id,
            num_prefetched_partitions,
            parallel_prefetch_requests,
            shuffle,
            tokenizer,
            None,  # log_path
            True,  # include_labels
        )
        self._sample_prob = sample_prob

    def __iter__(self) -> Generator:
        for transformed_tuple in OnlineDataset.__iter__(self):
            if self._sample_prob is not None:
                if random.random() < self._sample_prob:
                    yield transformed_tuple
            else:
                yield transformed_tuple
