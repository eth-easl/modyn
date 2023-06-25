from abc import ABC, abstractmethod

from sqlalchemy import Select


class AbstractPresamplingStragy(ABC):
    def __init__(
        self,
        config: dict,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
        tail_triggers: int,
        has_limit: bool,
        training_set_size_limit: int,
    ):
        self._config = config
        self._modyn_config = modyn_config
        self._pipeline_id = pipeline_id
        self._maximum_keys_in_memory = maximum_keys_in_memory
        self._tail_triggers = tail_triggers
        self._has_limit = has_limit
        self._training_set_size_limit = training_set_size_limit

    @abstractmethod
    def get_query_stmt(self, next_trigger_id: int) -> Select:
        raise NotImplementedError()
