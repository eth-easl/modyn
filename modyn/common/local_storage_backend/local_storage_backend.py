import ctypes
import logging
import os
import sys
import typing
from pathlib import Path
from sys import platform

import numpy as np
from modyn.utils import get_partition_for_worker
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)

NUMPY_HEADER_SIZE = 128


class ArrayWrapper:
    def __init__(self, array: np.ndarray, f_release: typing.Callable) -> None:
        self.array = array
        self.f_release = f_release

    def __len__(self) -> int:
        return self.array.size

    def __getitem__(self, key: typing.Any) -> typing.Any:
        return self.array.__getitem__(key)

    def __str__(self) -> str:
        return self.array.__str__()

    def __del__(self) -> None:
        self.f_release(self.array)

    def __eq__(self, other: typing.Any) -> typing.Any:
        return self.array == other

    def __ne__(self, other: typing.Any) -> typing.Any:
        return self.array == other

    @property
    def dtype(self) -> np.dtype:
        return self.array.dtype

    @property
    def shape(self) -> tuple:
        return self.array.shape


class LocalStorageBackend:
    """
    A trigger sample is a tuple of (sample_id, sample_weight) that is used to select a sample for a trigger.

    The sample_id is the id of the sample in the database. The sample_weight is the weight of the sample in the trigger.

    This class is used to store and retrieve trigger samples from the local file system. The trigger samples are stored
    in the directory specified in the modyn config file. The file name is the concatenation of the pipeline id, the
    trigger id and the partition id. The file contains one line per sample. Each line contains the sample id and the
    sample weight separated by a comma.
    """

    def _get_library_path(self) -> Path:
        if platform == "darwin":
            library_filename = "liblocal_storage_backend.dylib"
        else:
            library_filename = "liblocal_storage_backend.so"

        return self._get_build_path() / library_filename

    def _get_build_path(self) -> Path:
        return Path(__file__).parent.parent.parent.parent

    def _ensure_library_present(self) -> None:
        path = self._get_library_path()
        if not path.exists():
            raise RuntimeError(f"Cannot find TriggerSampleStorage library at {path}")

    def __init__(self, trigger_sample_directory="test_sample_storage") -> None:
        self._ensure_library_present()
        self.extension = ctypes.CDLL(str(self._get_library_path()))

        self.trigger_sample_directory = trigger_sample_directory
        if not Path(self.trigger_sample_directory).exists():
            Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created the trigger sample directory {self.trigger_sample_directory}.")
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

    def _parse_files(self, file_paths: list, data_lengths: list, data_offsets: list) -> None:
        """Read the trigger samples to multiple files.

        Args:
            file_path (str): File path to write to.
            trigger_samples (np.ndarray): List of trigger samples.
            data_lengths (list): List of
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

    def _write_files(self, file_paths: list, trigger_samples: np.ndarray, data_lengths: list) -> None:
        """Write the trigger samples to multiple files.

        Args:
            file_path (str): File path to write to.
            trigger_samples (np.ndarray): List of trigger samples.
            data_lengths (list): List of
        """

        data = np.asanyarray(trigger_samples)

        files = [ctypes.c_char_p(str(file_path.with_suffix(".nnpy")).encode("utf-8")) for file_path in file_paths]

        files_p = (ctypes.c_char_p * len(files))()
        headers_p = (ctypes.c_char_p * len(files))()
        data_lengths_p = (ctypes.c_int64 * len(data_lengths))()

        for i, _ in enumerate(files):
            files_p[i] = files[i]
            data_lengths_p[i] = data_lengths[i]

        self._write_files_impl(files_p, data, data_lengths_p, len(data_lengths))
