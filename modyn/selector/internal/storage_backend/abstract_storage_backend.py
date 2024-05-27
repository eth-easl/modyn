import logging
import os
import platform
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


class AbstractStorageBackend(ABC):
    def __init__(self, pipeline_id: int, modyn_config: dict, maximum_keys_in_memory: int):
        self._modyn_config = modyn_config
        self._maximum_keys_in_memory = maximum_keys_in_memory
        self._pipeline_id = pipeline_id

        raw_insertion_threads = modyn_config["selector"]["insertion_threads"]

        self._is_test = "PYTEST_CURRENT_TEST" in os.environ
        self._is_mac = platform.system() == "Darwin"

        if raw_insertion_threads <= 0 or (self._is_test and self._is_mac):
            self.insertion_threads = 1
        else:
            self.insertion_threads = raw_insertion_threads

        if self._maximum_keys_in_memory < 1:
            raise ValueError(f"Invalid setting for maximum_keys_in_memory: {self._maximum_keys_in_memory}")

    @abstractmethod
    def persist_samples(
        self, seen_in_trigger_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def get_available_labels(self, next_trigger_id: int, tail_triggers: Optional[int] = None) -> list[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_trigger_data(self, trigger_id: int) -> Iterable[tuple[list[int], dict[str, object]]]:
        raise NotImplementedError()

    @abstractmethod
    def get_data_since_trigger(
        self, smallest_included_trigger_id: int
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        raise NotImplementedError()

    @abstractmethod
    def get_all_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        raise NotImplementedError()
