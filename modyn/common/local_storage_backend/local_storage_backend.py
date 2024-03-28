import ctypes
import logging
import sys
import typing
from pathlib import Path
from sys import platform

import numpy as np
from numpy.ctypeslib import ndpointer

logger = logging.getLogger(__name__)

NUMPY_HEADER_SIZE = 128


class LocalStorageBackend:
    """
    This class is used to store and retrieve samples from the local file system. The samples are stored
    in the directory specified in the modyn config file. This class is only used for directly writing
    and reading the data by wrapping around CPP, the actual logic behind this backend is found in
    the storage backend folder.
    """

    def _get_library_path(self) -> Path:
        if platform == "darwin":
            library_filename = "liblocal_storage_backend.dylib"
        else:
            library_filename = "liblocal_storage_backend.so"

        return self._get_build_path() / library_filename

    def _get_build_path(self) -> Path:
        return Path(__file__).parent.parent.parent.parent / "libbuild"

    def _ensure_library_present(self) -> None:
        path = self._get_library_path()
        if not path.exists():
            raise RuntimeError(f"Cannot find LocalStorageBackend library at {path}")

    def __init__(self, local_storage_directory: typing.Union[str, Path]) -> None:
        self._ensure_library_present()
        self.extension = ctypes.CDLL(str(self._get_library_path()))

        self.local_storage_directory = local_storage_directory
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

    def _parse_files(self, file_paths: list, data_lengths: list, data_offsets: list) -> None:
        """Read data from multiple files. Reading supports offset and length per file.

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
