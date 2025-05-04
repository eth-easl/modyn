import ctypes
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from sys import platform
from typing import Any

import numpy as np
from numpy.ctypeslib import ndpointer

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.selector.internal.storage_backend import AbstractStorageBackend

logger = logging.getLogger(__name__)


class LocalStorageBackend(AbstractStorageBackend):
    """This class is used to store and retrieve samples from the local file
    system.

    The samples are stored in the directory specified in the modyn
    config file.
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        self._ensure_library_present()
        self.extension = ctypes.CDLL(str(self._get_library_path()))

        self.max_samples_in_file = self._modyn_config["selector"]["local_storage_max_samples_in_file"]

        self.local_storage_directory = self._modyn_config["selector"]["local_storage_directory"]
        if not Path(self.local_storage_directory).exists():
            Path(self.local_storage_directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created the local storage backend directory {self.local_storage_directory}.")
        if sys.maxsize < 2**63 - 1:
            raise RuntimeError("Modyn Selector Implementation requires a 64-bit system.")

        self._write_files_impl = self.extension.write_files
        self._write_files_impl.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ndpointer(flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
        ]
        self._write_files_impl.restype = None

        self._parse_files_impl = self.extension.parse_files
        self._parse_files_impl.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ndpointer(flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
        ]
        self._parse_files_impl.restype = None

    def _get_library_path(self) -> Path:
        if platform == "darwin":
            library_filename = "liblocal_storage_backend.dylib"
        else:
            library_filename = "liblocal_storage_backend.so"

        return self._get_build_path() / library_filename

    def _get_build_path(self) -> Path:
        return Path(__file__).parent.parent.parent.parent.parent.parent / "libbuild"

    def _ensure_library_present(self) -> None:
        path = self._get_library_path()
        if not path.exists():
            raise RuntimeError(f"Cannot find LocalStorageBackend library at {path}")

    def _parse_files(self, file_paths: list, data_lengths: list, data_offsets: list) -> np.ndarray:
        """Read data from multiple files. Reading supports offset and length
        per file.

        Args:
            file_paths (list): File paths to write to.
            data_lengths (list): Lengths of data to read from files.
            data_offsets (list): Offsets of data to read from files.
        """

        files = [ctypes.c_char_p(str(file_path.with_suffix(".nnpy")).encode("utf-8")) for file_path in file_paths]

        files_p = (ctypes.c_char_p * len(files))()
        data_lengths_p = (ctypes.c_int64 * len(data_lengths))()
        data_offsets_p = (ctypes.c_int64 * len(data_offsets))()

        total_length = sum(data_lengths)

        data = np.empty((total_length,), dtype=np.uint64)

        for i, _ in enumerate(files):
            files_p[i] = files[i]
            data_lengths_p[i] = data_lengths[i]
            data_offsets_p[i] = data_offsets[i]

        self._parse_files_impl(files_p, data, data_lengths_p, data_offsets_p, len(file_paths))
        return data

    def _write_files(self, file_paths: list, samples: np.ndarray, data_lengths: list) -> None:
        """Write data to multiple files.

        Args:
            file_paths (list): File paths to write to.
            samples (np.ndarray): Array of samples to write.
            data_lengths (list): Lengths of data to write to files.
        """

        data = np.asanyarray(samples)

        files = [ctypes.c_char_p(str(file_path.with_suffix(".nnpy")).encode("utf-8")) for file_path in file_paths]

        files_p = (ctypes.c_char_p * len(files))()
        data_lengths_p = (ctypes.c_int64 * len(data_lengths))()

        for i, _ in enumerate(files):
            files_p[i] = files[i]
            data_lengths_p[i] = data_lengths[i]

        self._write_files_impl(files_p, data, data_lengths_p, len(data_lengths))

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
            yield self._parse_files(filenames, data_lengths, data_offsets).tolist(), {}

    # pylint: disable=too-many-locals

    def persist_samples(
        self, seen_in_trigger_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, Any]:
        log = {}
        swt = Stopwatch()
        swt.start("persist_samples_time")
        root = self._modyn_config["selector"]["local_storage_directory"]
        trigger_folder = Path(root) / str(self._pipeline_id) / str(seen_in_trigger_id)
        Path(trigger_folder).mkdir(parents=True, exist_ok=True)

        keys_array = np.array(keys, dtype=np.uint64)
        labels_array = set(labels)

        (trigger_folder / "labels").mkdir(parents=True, exist_ok=True)

        for label in labels_array:
            with open(trigger_folder / "labels" / str(label), "w", encoding="utf-8"):
                continue

        existing_count = len(list(trigger_folder.glob("*")))

        data_lengths = [self.max_samples_in_file] * (len(keys_array) // self.max_samples_in_file)

        if sum(data_lengths) < len(keys_array):
            data_lengths.append(len(keys_array) - sum(data_lengths))

        file_paths = []
        for data_length in data_lengths:
            file_paths.append(trigger_folder / f"{existing_count}_{data_length}.npy")
            existing_count += 1

        self._write_files(file_paths, keys_array, data_lengths)
        log["persist_samples_time"] = swt.stop()
        return log

    def get_available_labels(self, next_trigger_id: int, tail_triggers: int | None = None) -> list[int]:
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

    def _get_data_from_storage(
        self, selector_keys: list[int], dataset_id: str
    ) -> Iterator[tuple[list[int], list[bytes], list[int], list[bytes], int]]:
        """
        Retrieve full sample data from storage given a list of keys.
        """
        raise NotImplementedError()
