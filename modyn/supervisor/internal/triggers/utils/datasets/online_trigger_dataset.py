import logging
import random
from collections.abc import Generator

from torch.utils.data import IterableDataset

from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset

logger = logging.getLogger(__name__)


# TODO(#275): inherit common abstraction of dataset
class OnlineTriggerDataset(OnlineDataset, IterableDataset):
    """The OnlineTriggerDataset is a wrapper around OnlineDataset in
    trainer_server.

    It uses logic in OnlineDataset obtain samples by trigger_id. It
    supports random sampling to reduce the number of samples if the
    sample_prob is provided. Random sampling is needed for example in
    DataDriftTrigger to reduce the number of samples processed in data
    drift detection in case there are too many untriggered samples.
    """

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
        shuffle: bool,
        tokenizer: str | None = None,
        sample_prob: float | None = None,
    ):
        OnlineDataset.__init__(
            self,
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
            shuffle,
            tokenizer,
            None,  # log_path
            True,  # include_labels
        )
        self._sample_prob = sample_prob

    def __iter__(self) -> Generator:
        for transformed_tuple in OnlineDataset.__iter__(self):
            if self._sample_prob is not None:
                prob = random.random()
                if prob < self._sample_prob:
                    yield transformed_tuple
            else:
                yield transformed_tuple
