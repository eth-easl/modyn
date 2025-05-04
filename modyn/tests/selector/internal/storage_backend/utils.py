from collections.abc import Iterable, Iterator
from typing import Any

from modyn.selector.internal.storage_backend import AbstractStorageBackend


class MockStorageBackend(AbstractStorageBackend):
    # pylint: disable=super-init-not-called
    def __init__(self, pipeline_id: int, modyn_config: dict, maximum_keys_in_memory: int):
        self.insertion_threads = 1
        return

    def persist_samples(
        self, seen_in_trigger_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, Any]:
        pass

    def get_available_labels(self, next_trigger_id: int, tail_triggers: int | None = None) -> list[int]:
        pass

    def get_trigger_data(self, trigger_id: int) -> Iterable[tuple[list[int], dict[str, object]]]:
        pass

    def get_data_since_trigger(
        self, smallest_included_trigger_id: int
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        pass

    def get_all_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        pass

    def _get_data_from_storage(
        self, selector_keys: list[int], worker_id: int | None = None
    ) -> Iterator[tuple[list[int], list[bytes], list[int] | list[bytes], int]]:
        pass
