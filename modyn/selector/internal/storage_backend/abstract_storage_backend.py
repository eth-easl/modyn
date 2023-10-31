import logging
import os
import platform
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class AbstractStorageBackend(ABC):
    def __init__(self, pipeline_id: int, modyn_config: dict, maximum_keys_in_memory: int):
        self._modyn_config = modyn_config
        self._maximum_keys_in_memory = maximum_keys_in_memory
        self._pipeline_id = pipeline_id

        self._insertion_threads = modyn_config["selector"]["insertion_threads"]
        self._is_test = "PYTEST_CURRENT_TEST" in os.environ
        self._is_mac = platform.system() == "Darwin"
        self._disable_mt = self._insertion_threads <= 0

        if self._maximum_keys_in_memory < 1:
            raise ValueError(f"Invalid setting for maximum_keys_in_memory: {self._maximum_keys_in_memory}")

    @abstractmethod
    def persist_samples(
        self, seen_in_trigger_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def get_available_labels(self) -> list[int]:
        raise NotImplementedError()
