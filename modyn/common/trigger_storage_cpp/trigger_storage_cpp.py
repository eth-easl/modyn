import ctypes
import logging
import os
import sys
from pathlib import Path
from sys import platform

import numpy as np
from numpy.ctypeslib import ndpointer

from .utils import get_partition_for_worker

logger = logging.getLogger(__name__)


class ArrayWrapper:
    def __init__(self, array, f_release):
        self.array = array
        self.f_release = f_release

    def __len__(self):
        return self.array.size

    def __getitem__(self, key):
        return self.array.__getitem__(key)

    def __str__(self):
        return self.array.__str__()

    def __del__(self):
        self.f_release(self.array)

    def __eq__(self, other):
        return self.array == other

    def __ne__(self, other):
        return self.array == other

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def shape(self):
        return self.array.shape


class TriggerStorageCPP:
    """
    A trigger sample is a tuple of (sample_id, sample_weight) that is used to select a sample for a trigger.

    The sample_id is the id of the sample in the database. The sample_weight is the weight of the sample in the trigger.

    This class is used to store and retrieve trigger samples from the local file system. The trigger samples are stored
    in the directory specified in the modyn config file. The file name is the concatenation of the pipeline id, the
    trigger id and the partition id. The file contains one line per sample. Each line contains the sample id and the
    sample weight separated by a comma.
    """

    @staticmethod
    def _get_library_path() -> Path:
        if platform == "darwin":
            library_filename = "libtrigger_storage_cpp.dylib"
        else:
            library_filename = "libtrigger_storage_cpp.so"

        return TriggerStorageCPP._get_build_path() / library_filename

    @staticmethod
    def _get_build_path() -> Path:
        return Path(__file__).parent.parent.parent.parent / "libbuild"

    @staticmethod
    def _ensure_library_present() -> None:
        path = TriggerStorageCPP._get_library_path()
        if not path.exists():
            raise RuntimeError(f"Cannot find {TriggerStorageCPP.__name__} library at {path}")

    def __init__(self, trigger_sample_directory: str = "sample_dir") -> None:
        TriggerStorageCPP._ensure_library_present()
        self.extension = ctypes.CDLL(str(TriggerStorageCPP._get_library_path()))

        self.trigger_sample_directory = trigger_sample_directory
        if not Path(self.trigger_sample_directory).exists():
            Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created the trigger sample directory {self.trigger_sample_directory}.")
        if sys.maxsize < 2**63 - 1:
            raise RuntimeError("Modyn Selector Implementation requires a 64-bit system.")

        # We define a return object here that can infer its size when being parsed by ctypes
        self.data_pointer = ndpointer(dtype=[("f0", "<i8"), ("f1", "<f8")], ndim=1, shape=(1,), flags="C_CONTIGUOUS")
        self.data_pointer._shape_ = property(lambda self: self.shape_val[0])

        self._get_num_samples_in_file_impl = self.extension.get_num_samples_in_file
        self._get_num_samples_in_file_impl.argtypes = [ctypes.POINTER(ctypes.c_char)]
        self._get_num_samples_in_file_impl.restype = ctypes.c_uint64

        self._write_file_impl = self.extension.write_file
        self._write_file_impl.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ndpointer(flags="C_CONTIGUOUS"),
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_uint64,
        ]
        self._write_file_impl.restype = None

        self._write_files_impl = self.extension.write_files
        self._write_files_impl.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ndpointer(flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self._write_files_impl.restype = None

        self._get_all_samples_impl = self.extension.get_all_samples
        self._get_all_samples_impl.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_char),
        ]
        self._get_all_samples_impl.restype = self.data_pointer

        self._get_worker_samples_impl = self.extension.get_worker_samples
        self._get_worker_samples_impl.argtypes = [
            ctypes.POINTER(ctypes.c_char),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        self._get_worker_samples_impl.restype = self.data_pointer

        self._parse_file_impl = self.extension.parse_file
        self._parse_file_impl.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_uint64)]
        self._parse_file_impl.restype = self.data_pointer

        self._release_data_impl = self.extension.release_data
        self._release_data_impl.argtypes = [ndpointer(dtype=[("f0", "<i8"), ("f1", "<f8")], flags="C_CONTIGUOUS")]
        self._release_data_impl.restype = None

    def get_trigger_samples(
        self,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
        retrieval_worker_id: int = -1,
        total_retrieval_workers: int = -1,
        num_samples_trigger_partition: int = -1,
    ) -> ArrayWrapper:
        """
        Return the trigger samples for the given pipeline id, trigger id and partition id.

        If the retrieval worker id and the total retrieval workers are negative, then we are not using the parallel
        retrieval of samples. In this case, we just return all the samples.

        If the retrieval worker id and the total retrieval workers are positive, then we are using the parallel
        retrieval of samples. In this case, we return the samples that are assigned to the retrieval worker.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :param retrieval_worker_id: the id of the retrieval worker
        :param total_retrieval_workers: the total number of retrieval workers
        :param num_samples_trigger_partition: the total number of samples per trigger and partition
        :return: the trigger samples
        """
        if not Path(self.trigger_sample_directory).exists():
            raise FileNotFoundError(f"The trigger sample directory {self.trigger_sample_directory} does not exist.")
        assert (retrieval_worker_id >= 0 and total_retrieval_workers >= 0) or (
            retrieval_worker_id < 0 and total_retrieval_workers < 2
        ), "Either both or none of the retrieval worker id must be negative and \
            the total retrieval workers must be smaller than 2."
        if retrieval_worker_id < 0 and total_retrieval_workers < 2:
            return self._get_all_samples(pipeline_id, trigger_id, partition_id)
        assert num_samples_trigger_partition > 0, "The number of samples per trigger must be positive."
        return self._get_worker_samples(
            pipeline_id,
            trigger_id,
            partition_id,
            retrieval_worker_id,
            total_retrieval_workers,
            num_samples_trigger_partition,
        )

    def _get_worker_samples(
        self,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
        retrieval_worker_id: int,
        total_retrieval_workers: int,
        num_samples_trigger_partition: int,
    ) -> ArrayWrapper:
        """
        Return the trigger samples for the given pipeline id, trigger id and partition id that are assigned to the
        retrieval worker.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :param retrieval_worker_id: the id of the retrieval worker
        :param total_retrieval_workers: the total number of retrieval workers
        :param num_samples_trigger_partition: the total number of samples per trigger and partition
        :return: the trigger samples
        """
        start_index, worker_subset_size = get_partition_for_worker(
            retrieval_worker_id, total_retrieval_workers, num_samples_trigger_partition
        )

        folder = ctypes.c_char_p(str(self.trigger_sample_directory).encode("utf-8"))
        size = (ctypes.c_uint64 * 1)()
        self.data_pointer.shape_val = size
        pattern = ctypes.c_char_p(f"{pipeline_id}_{trigger_id}_{partition_id}_".encode("utf-8"))

        data = self._get_worker_samples_impl(folder, size, pattern, start_index, worker_subset_size).reshape(-1)
        result = ArrayWrapper(data, self._release_data_impl)
        return result

    def _get_all_samples(self, pipeline_id: int, trigger_id: int, partition_id: int) -> ArrayWrapper:
        """
        Return all the samples for the given pipeline id, trigger id and partition id.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :return: the trigger samples
        """

        folder = ctypes.c_char_p(str(self.trigger_sample_directory).encode("utf-8"))
        size = (ctypes.c_uint64 * 1)()
        self.data_pointer.shape_val = size
        pattern = ctypes.c_char_p(f"{pipeline_id}_{trigger_id}_{partition_id}_".encode("utf-8"))

        data = self._get_all_samples_impl(folder, size, pattern).reshape(-1)
        result = ArrayWrapper(data, self._release_data_impl)
        return result

    def save_trigger_sample(
        self, pipeline_id: int, trigger_id: int, partition_id: int, trigger_samples: np.ndarray, insertion_id: int
    ) -> None:
        """
        Save the trigger samples for the given pipeline id, trigger id and partition id.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :param trigger_samples: the trigger samples
        :param insertion_id: the id of the insertion
        """
        if trigger_samples.dtype != np.dtype("i8,f8"):
            raise ValueError(f"Unexpected dtype: {trigger_samples.dtype}\nExpected: {np.dtype('i8,f8')}")

        Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)

        samples_file = Path(self.trigger_sample_directory) / f"{pipeline_id}_{trigger_id}_{partition_id}_{insertion_id}"

        assert not Path(samples_file).exists(), (
            f"Trigger samples file {samples_file} already exists. " f"Please delete it if you want to overwrite it."
        )

        self._write_file(samples_file, trigger_samples)

    def save_trigger_samples(
        self, pipeline_id: int, trigger_id: int, partition_id: int, trigger_samples: np.ndarray, data_lengths: list
    ) -> None:
        """
        Save the trigger samples for the given pipeline id, trigger id and partition id
        to multiple files.

        :param pipeline_id: the id of the pipeline
        :param trigger_id: the id of the trigger
        :param partition_id: the id of the partition
        :param trigger_samples: the trigger samples
        :param data_lengths: the lengths of subarrays to write to files
        """
        if trigger_samples.dtype != np.dtype("i8,f8"):
            raise ValueError(f"Unexpected dtype: {trigger_samples.dtype}\nExpected: {np.dtype('i8,f8')}")

        Path(self.trigger_sample_directory).mkdir(parents=True, exist_ok=True)

        samples_files = []
        for i in range(len(data_lengths)):
            samples_files.append(Path(self.trigger_sample_directory) / f"{pipeline_id}_{trigger_id}_{partition_id}_{i}")
            assert not Path(samples_files[i]).exists(), (
                f"Trigger samples file {samples_files[i]} already exists. "
                f"Please delete it if you want to overwrite it."
            )

        self._write_files(samples_files, trigger_samples, data_lengths)

    def get_file_path(self, pipeline_id: int, trigger_id: int, partition_id: int, worker_id: int) -> Path:
        return Path(self.trigger_sample_directory) / f"{pipeline_id}_{trigger_id}_{partition_id}_{worker_id}.npy"

    def _get_files_for_trigger(self, pipeline_id: int, trigger_id: int) -> list[str]:
        # here we filter the files belonging to the given pipeline and trigger

        return list(
            filter(
                lambda file: file.startswith(f"{pipeline_id}_{trigger_id}_"), os.listdir(self.trigger_sample_directory)
            )
        )

    def get_trigger_num_data_partitions(self, pipeline_id: int, trigger_id: int) -> int:
        # each file follows the structure {pipeline_id}_{trigger_id}_{partition_id}_{worker_id}

        this_trigger_files = self._get_files_for_trigger(pipeline_id, trigger_id)

        # then we count how many partitions we have (not just len(this_trigger_partitions) since there could be
        # multiple workers for each partition
        return len(set(file.split("_")[2] for file in this_trigger_files))

    def clean_trigger_data(self, pipeline_id: int, trigger_id: int) -> None:
        # remove all the files belonging to the given pipeline and trigger

        if os.path.isdir(self.trigger_sample_directory):
            this_trigger_files = self._get_files_for_trigger(pipeline_id, trigger_id)

            for file in this_trigger_files:
                os.remove(os.path.join(self.trigger_sample_directory, file))

    def _parse_file(self, file_path: Path) -> np.ndarray:
        """Parse the given file and return the samples.

        Args:
            file_path (str): File path to parse.

        Returns:
            np.ndarray: List of trigger samples.
        """

        file = ctypes.c_char_p(str(file_path).encode("utf-8"))
        size = (ctypes.c_uint64 * 1)()
        self.data_pointer.shape_val = size

        data = self._parse_file_impl(file, size).reshape(-1)
        result = ArrayWrapper(data, self._release_data_impl)

        return result

    def _get_num_samples_in_file(self, file_path: Path) -> int:
        """Get the number of samples in the given file.

        Args:
            file_path (str): File path to parse.
        """

        file = ctypes.c_char_p(str(file_path).encode("utf-8"))
        return self._get_num_samples_in_file_impl(file)

    def _write_file(self, file_path: list, trigger_samples: list) -> None:
        """Write the trigger samples to the given file.

        Args:
            file_path (str): File path to write to.
            trigger_samples (np.ndarray): List of trigger samples.
        """

        data = np.asanyarray(trigger_samples)

        header = self._build_array_header(np.lib.format.header_data_from_array_1_0(data))

        file = ctypes.c_char_p(str(file_path.with_suffix(".npy")).encode("utf-8"))
        header = ctypes.c_char_p(header)

        self._write_file_impl(file, data, 0, len(data), header, 128)

    def _write_files(self, file_paths: Path, trigger_samples: np.ndarray, data_lengths: list) -> None:
        """Write the trigger samples to multiple files.

        Args:
            file_path (str): File path to write to.
            trigger_samples (np.ndarray): List of trigger samples.
            data_lengths (list): List of
        """

        data = np.asanyarray(trigger_samples)

        raw_headers = []
        length_sum = 0

        for data_length in data_lengths:
            raw_headers.append(
                self._build_array_header(
                    np.lib.format.header_data_from_array_1_0(data[length_sum : length_sum + data_length])
                )
            )
            length_sum += data_length

        files = [ctypes.c_char_p(str(file_path.with_suffix(".npy")).encode("utf-8")) for file_path in file_paths]
        headers = [ctypes.c_char_p(header) for header in raw_headers]

        files_p = (ctypes.c_char_p * len(files))()
        headers_p = (ctypes.c_char_p * len(files))()
        data_lengths_p = (ctypes.c_uint64 * len(data_lengths))()

        for i, _ in enumerate(files):
            files_p[i] = files[i]
            headers_p[i] = headers[i]
            data_lengths_p[i] = data_lengths[i]

        self._write_files_impl.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ndpointer(flags="C_CONTIGUOUS"),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]

        self._write_files_impl(files_p, data, data_lengths_p, headers_p, 128, len(data_lengths))

    def _build_array_header(self, d):
        """Write the header for an array and returns the version used

        Parameters
        ----------
        fp : filelike object
        d : dict
            This has the appropriate entries for writing its string representation
            to the header of the file.
        version : tuple or None
            None means use oldest that works. Providing an explicit version will
            raise a ValueError if the format does not allow saving this data.
            Default: None
        """
        header = ["{"]
        for key, value in sorted(d.items()):
            # Need to use repr here, since we eval these when reading
            header.append(f"'{key}': {repr(value)}, ")
        header.append("}")
        header = "".join(header)

        # Add some spare space so that the array header can be modified in-place
        # when changing the array size, e.g. when growing it by appending data at
        # the end.
        shape = d["shape"]
        growth_axis_max_digits = 21
        header += " " * ((growth_axis_max_digits - len(repr(shape[0]))) if len(shape) > 0 else 0)

        header = np.lib.format._wrap_header_guess_version(header)
        return header