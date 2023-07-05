from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy import Select


class AbstractPresamplingStrategy(ABC):
    def __init__(self, presampling_config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        self.modyn_config = modyn_config
        self.pipeline_id = pipeline_id
        self.maximum_keys_in_memory = maximum_keys_in_memory

        if "ratio" not in presampling_config:
            raise ValueError("Please specify the presampling ratio.")
        self.presampling_ratio = presampling_config["ratio"]

        if not (0 < self.presampling_ratio <= 100) or not isinstance(self.presampling_ratio, int):
            raise ValueError("Presampling ratio must be an integer in range (0,100]")

        self.requires_trigger_dataset_size = False

    @abstractmethod
    def get_presampling_query(self, next_trigger_id: int, tail_triggers: Optional[int], limit: Optional[int],
                              trigger_dataset_size: Optional[int], requires_samples_ordered_by_label: bool) -> Select:
        raise NotImplementedError()

    def get_target_size(self, trigger_dataset_size: int, limit: Optional[int]) -> int:
        assert trigger_dataset_size >= 0
        target_presampling = int(trigger_dataset_size * self.presampling_ratio / 100)

        if limit is not None:
            assert limit >= 0
            target_size = min(limit, target_presampling)
        else:
            target_size = target_presampling

        return target_size
