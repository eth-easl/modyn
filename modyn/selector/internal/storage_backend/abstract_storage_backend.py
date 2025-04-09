import logging
import os
import platform
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

logger = logging.getLogger(__name__)


class AbstractStorageBackend(ABC):
    def __init__(self, pipeline_id: int, modyn_config: dict, maximum_keys_in_memory: int) -> None:
        self._modyn_config = modyn_config
        self._maximum_keys_in_memory = maximum_keys_in_memory
        self._pipeline_id = pipeline_id
        self._storagestub = None
        storage = modyn_config.get(
            "storage", {}
        )  # Storage is always part of config but this prevents having to rwritte all the tests.
        self._storage_address = (
            f"{storage.get('hostname')}:{storage.get('port')}"
            if storage.get("hostname") and storage.get("port")
            else None
        )

        raw_insertion_threads = modyn_config["selector"]["insertion_threads"]

        is_test = "PYTEST_CURRENT_TEST" in os.environ
        is_mac = platform.system() == "Darwin"

        if raw_insertion_threads <= 0 or (is_test and is_mac):
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
    def get_available_labels(self, next_trigger_id: int, tail_triggers: int | None = None) -> list[int]:
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

    @abstractmethod
    def _get_data_from_storage(
        self,
        selector_keys: list[int],
        dataset_it: str,
    ) -> Iterator[tuple[list[int], list[bytes], list[int], list[bytes], int]]:
        """
        Retrieve full sample data from storage given a list of keys.
        """
        raise NotImplementedError()
