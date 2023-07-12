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
        )

        self.filtered_label = None

    def _get_data_tuple(self, key: int, sample: bytes, label: int, weight: Optional[float]) -> Optional[Tuple]:
        assert self._uses_weights is not None

        if self.filtered_label != label:
            return None

        # mypy complains here because _transform has unknown type, which is ok
        if self._uses_weights:
            return key, self._transform(sample), label, weight  # type: ignore
        return key, self._transform(sample), label  # type: ignore
