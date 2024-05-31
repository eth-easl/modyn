from abc import ABC, abstractmethod
from typing import Optional


class AbstractKeySource(ABC):
    def __init__(self, pipeline_id: int, trigger_id: int) -> None:
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id

    @abstractmethod
    def get_keys_and_weights(
        self, worker_id: int, partition_id: int, shuffle: bool
    ) -> tuple[list[int], Optional[list[float]]]:
        raise NotImplementedError()

    @abstractmethod
    def get_num_data_partitions(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def uses_weights(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def init_worker(self) -> None:
        # init connections when pytorch worker is created
        raise NotImplementedError()

    @abstractmethod
    def end_of_trigger_cleaning(self) -> None:
        # remove temporary files when the trigger ends
        raise NotImplementedError()
