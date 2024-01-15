import logging

import numpy as np
from pathlib import Path
from typing import Any, Iterable, Optional

from modyn.selector.internal.storage_backend import AbstractStorageBackend
from modyn.common.local_storage_backend import LocalStorageBackend as LocalStorageBackendCPP

logger = logging.getLogger(__name__)

MAX_SAMPLES_IN_FILE = 1000000


class LocalStorageBackend(AbstractStorageBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cpp = LocalStorageBackendCPP()

    def _get_data(
        self, smallest_included_trigger_id: int, single_trigger: bool
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        root = self._modyn_config["selector"]["local_storage_directory"]

        filenames = []
        data_lengths = []
        data_offsets = []

        read_blocks = []
        block_size_left = self._maximum_keys_in_memory

        for trigger_folder in (Path(root) / str(self._pipeline_id)).glob("*"):
            if (int(trigger_folder.name) < smallest_included_trigger_id) or (
                single_trigger and (int(trigger_folder.name) > smallest_included_trigger_id)
            ):
                continue

            for file in trigger_folder.glob("*"):
                if file.name == "labels":
                    continue
                samples_in_file = int(file.stem.split("_")[-1])
                offset_in_file = 0

                while samples_in_file >= block_size_left:
                    filenames.append(file)
                    data_lengths.append(block_size_left)
                    data_offsets.append(offset_in_file)

                    read_blocks.append((filenames, data_lengths, data_offsets))

                    filenames = []
                    data_lengths = []
                    data_offsets = []

                    samples_in_file -= block_size_left
                    offset_in_file += block_size_left
                    block_size_left = self._maximum_keys_in_memory

                if samples_in_file == 0:
                    continue

                block_size_left -= samples_in_file

                filenames.append(file)
                data_lengths.append(samples_in_file)
                data_offsets.append(offset_in_file)

        if filenames:
            read_blocks.append((filenames, data_lengths, data_offsets))

        for filenames, data_lengths, data_offsets in read_blocks:
            yield self._cpp._parse_files(filenames, data_lengths, data_offsets), dict()

    def persist_samples(
        self, seen_in_trigger_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, Any]:
        root = self._modyn_config["selector"]["local_storage_directory"]
        trigger_folder = Path(root) / str(self._pipeline_id) / str(seen_in_trigger_id)
        Path(trigger_folder).mkdir(parents=True, exist_ok=True)

        keys_array = np.array(keys, dtype=np.uint64)
        labels_array = set(labels)

        (trigger_folder / "labels").mkdir(parents=True, exist_ok=True)

        for label in labels_array:
            with open(trigger_folder / "labels" / str(label), "w"):
                continue

        existing_count = len(list(trigger_folder.glob("*")))

        data_lengths = [MAX_SAMPLES_IN_FILE] * (len(keys_array) // MAX_SAMPLES_IN_FILE)

        if sum(data_lengths) < len(keys_array):
            data_lengths.append(len(keys_array) - sum(data_lengths))

        file_paths = []
        for data_length in data_lengths:
            file_paths.append(trigger_folder / f"{existing_count}_{data_length}.npy")
            existing_count += 1

        self._cpp._write_files(file_paths, keys_array, data_lengths)

    def get_available_labels(self, next_trigger_id: int, tail_triggers: Optional[int] = None) -> list[int]:
        root = self._modyn_config["selector"]["local_storage_directory"]

        available_labels = set()

        for trigger_folder in (Path(root) / str(self._pipeline_id)).glob("*"):
            if ((tail_triggers is not None) and (int(trigger_folder.name) < next_trigger_id - tail_triggers - 1)) or (
                int(trigger_folder.name) >= next_trigger_id
            ):
                continue

            available_labels |= set(map(lambda x: int(x.name), (trigger_folder / "labels").glob("*")))

        return list(available_labels)

    def get_trigger_data(self, trigger_id: int) -> Iterable[tuple[list[int], dict[str, object]]]:
        return self._get_data(trigger_id, True)

    def get_data_since_trigger(
        self, smallest_included_trigger_id: int
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        return self._get_data(smallest_included_trigger_id, False)

    def get_all_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        return self.get_data_since_trigger(-1)
