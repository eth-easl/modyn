import logging
import random
from typing import Generator, Optional

from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


# TODO(#275): inherit common abstraction of dataset
class OnlineTriggerDataset(IterableDataset):
    # pylint: disable=too-many-instance-attributes, abstract-method
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
        tokenizer: Optional[str] = None,
        sample_prob: Optional[float] = None,
    ):
        self.online_dataset = OnlineDataset(
            pipeline_id,
            trigger_id,
            dataset_id,
            bytes_parser,
            serialized_transforms,
            storage_address,
            selector_address,
            training_id,
            num_prefetched_partitions,
            parallel_prefetch_requests,
            tokenizer,
            None,
        )
        self._sample_prob = sample_prob

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements

    def __iter__(self) -> Generator:
        for transformed_tuple in self.online_dataset:
            if self._sample_prob is not None:
                prob = random.random()
                if prob < self._sample_prob:
                    yield transformed_tuple
            else:
                yield transformed_tuple
