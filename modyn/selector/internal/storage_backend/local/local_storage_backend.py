import logging
from typing import Any, Iterable, Optional

from modyn.selector.internal.storage_backend import AbstractStorageBackend

logger = logging.getLogger(__name__)


class LocalStorageBackend(AbstractStorageBackend):
    def persist_samples(
        self, seen_in_trigger_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, Any]:
        raise NotImplementedError()

    def get_available_labels(self, next_trigger_id: int, tail_triggers: Optional[int] = None) -> list[int]:
        raise NotImplementedError()

    def get_trigger_data(self, trigger_id: int) -> Iterable[tuple[list[int], dict[str, object]]]:
        raise NotImplementedError()

    def get_data_since_trigger(
        self, smallest_included_trigger_id: int
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        raise NotImplementedError()

    def get_all_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        raise NotImplementedError()
