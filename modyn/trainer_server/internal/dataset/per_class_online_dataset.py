import logging
from typing import Optional, Tuple

from modyn.trainer_server.internal.dataset.online_dataset import OnlineDataset

logger = logging.getLogger(__name__)


class PerClassOnlineDataset(OnlineDataset):
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
        initial_filtered_label: int,
        num_prefetched_partitions: int,
        parallel_prefetch_requests: int,
        tokenizer: Optional[str],
    ):
        super().__init__(
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
        assert initial_filtered_label is not None
        self.filtered_label = initial_filtered_label

    def _get_transformed_data_tuple(
        self, key: int, sample: bytes, label: int, weight: Optional[float]
    ) -> Optional[Tuple]:
        assert self.filtered_label is not None

        if self.filtered_label != label:
            return None
        return super()._get_transformed_data_tuple(key, sample, label, weight)  # type: ignore
