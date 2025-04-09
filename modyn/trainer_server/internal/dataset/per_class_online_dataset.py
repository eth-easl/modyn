import logging
import pathlib

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
        shuffle: bool,
        tokenizer: str | None,
        include_labels: bool = True,
        *,
        bytes_parser_target: str | None = None,
        serialized_transforms_target: list[str] | None = None,
        log_path: pathlib.Path | None = None,
    ):
        # Pass all required arguments to the parent __init__
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
            shuffle,
            tokenizer,
            log_path,
            include_labels,
            bytes_parser_target,
            serialized_transforms_target,
        )
        self.filtered_label = initial_filtered_label

    def _get_transformed_data_tuple(
        self,
        key: int,
        sample: memoryview,
        label: int | None = None,  # type: ignore
        weight: float | None | memoryview = None,
    ) -> tuple | None:
        if self.filtered_label != label:
            return None
        return super()._get_transformed_data_tuple(key, sample, label, weight)
