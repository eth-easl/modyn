from abc import ABC, abstractmethod
from typing import Optional

from sqlalchemy import Select


class AbstractPresamplingStrategy(ABC):
    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        self._config = config
        self.modyn_config = modyn_config
        self.pipeline_id = pipeline_id
        self.maximum_keys_in_memory = maximum_keys_in_memory

    @abstractmethod
    def get_presampling_query(
        self,
        next_trigger_id: int,
        tail_triggers: Optional[int],
        limit: Optional[int],
        trigger_dataset_size: Optional[int],
    ) -> Select:
        raise NotImplementedError()

    @abstractmethod
    def requires_trigger_dataset_size(
        self,
    ) -> bool:
        raise NotImplementedError()
