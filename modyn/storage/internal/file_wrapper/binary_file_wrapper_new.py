"""Binary file wrapper."""

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType
from sys import platform
from pathlib import Path
import logging
import subprocess
import ctypes

logger = logging.getLogger(__name__)

class IntVector(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_int)),
                ("size", ctypes.c_size_t)]

class BinaryFileWrapperNew(AbstractFileWrapper):
    """Binary file wrapper.

    Binary files store raw sample data in a row-oriented format. One file can contain multiple samples.
    This wrapper requires that each samples should start with the label followed by its set of features.
    Each sample should also have a fixed overall width (in bytes) and a fixed width for the label,
    both of which should be provided in the config. The file wrapper is able to read samples by
    offsetting the required number of bytes.
    """

    def __init__(
        self,
        file_path: str,
        file_wrapper_config: dict,
        filesystem_wrapper: AbstractFileSystemWrapper,
    ):
        """Init binary file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
            filesystem_wrapper (AbstractFileSystemWrapper): File system wrapper to abstract storage of the file

        Raises:
            ValueError: If the file has the wrong file extension
            ValueError: If the file does not contain an exact number of samples of given size
        """
        super().__init__(file_path, file_wrapper_config, filesystem_wrapper)

        # Load the binary file wrapper library
        BinaryFileWrapperNew.__ensure_binary_file_wrapper_present()
        binary_file_wrapper_path = BinaryFileWrapperNew.__get_binary_file_wrapper_path()
        self.binary_file_wrapper_cpp = ctypes.cdll.LoadLibrary(str(binary_file_wrapper_path))

        self.file_wrapper_type = FileWrapperType.BinaryFileWrapper
        self.byteorder = file_wrapper_config["byteorder"]
        self._mode = 0 # 0 for normal mode (non-local filesystem), 1 for local filesystem (for faster read/write native in c++)

        self.record_size = file_wrapper_config["record_size"]
        self.label_size = file_wrapper_config["label_size"]
        if self.record_size - self.label_size < 1:
            raise ValueError("Each record must have at least 1 byte of data other than the label.")

        self._validate_file_extension()
        self.file_size = self.filesystem_wrapper.get_size(self.file_path)
        if self.file_size % self.record_size != 0:
            raise ValueError("File does not contain exact number of records of size " + str(self.record_size))
    
        if self.filesystem_wrapper.filesystem_wrapper_type == FilesystemWrapperType.LocalFilesystemWrapper:
            self._mode = 1

    def ensure_binary_file_wrapper_compiled(self):
        pass

    @staticmethod
    def __get_binary_file_wrapper_path():
        if platform == "darwin":
            binary_file_wrapper_filename = "libbinary_file_wrapper.dylib"
        else:
            binary_file_wrapper_filename = "libbinary_file_wrapper.so"
        return BinaryFileWrapperNew.__get_build_path() / binary_file_wrapper_filename

    @staticmethod
    def __get_build_path():
        return Path(__file__).parent / "binary_file_wrapper" / "build"

    @staticmethod
    def __ensure_binary_file_wrapper_present():
        if not BinaryFileWrapperNew.__get_binary_file_wrapper_path().exists():
            logger.info('Unweighted not built yet. Building...')
            build_path = BinaryFileWrapperNew.__get_build_path()
            # Execute `cmake ..` in build folder
            subprocess.run(['cmake', '..'], check=True, cwd=build_path)
            # Execute `make reduce` in build folder
            subprocess.run(['make', '-j8', 'binary_file_wrapper'], check=True, cwd=build_path)

    def _validate_file_extension(self) -> None:
        """Validates the file extension as bin

        Raises:
            ValueError: File has wrong file extension
        """
        if not self.file_path.endswith(".bin"):
            raise ValueError("File has wrong file extension.")

    def _validate_request_indices(self, total_samples: int, indices: list) -> None:
        """Validates if the requested indices are in the range of total number of samples
            in the file

        Args:
            total_samples: Total number of samples in the file
            indices (list): List of indices of the required samples

        Raises:
            IndexError: If the index is out of bounds
        """
        # TODO: Call this function in cpp
        indices_ptr = IntVector((ctypes.c_int * len(indices))(*indices), len(indices))
        total_samples_ptr = ctypes.c_int(total_samples)
        result_ptr = self.binary_file_wrapper_cpp.validate_request_indices(ctypes.byref(indices_ptr), total_samples_ptr)

        if result_ptr == 0:
            raise IndexError("Indices are out of range. Indices should be between 0 and " + str(total_samples))

    def get_number_of_samples(self) -> int:
        """Get number of samples in file.

        Returns:
            int: Number of samples in file
        """
        return int(self.file_size / self.record_size)

    def get_label(self, index: int) -> int:
        """Get the label of the sample at the given index.

        Args:
            index (int): Index

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            int: Label for the sample
        """
        if self._mode == 1:
            return self.get_label_native_cpp(index)
        else:
            return self.get_label_cpp(index)

    def get_label_native_cpp(self, index: int) -> int:
        """Get the label of the sample at the given index.

        Args:
            index (int): Index

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            int: Label for the sample
        """
        index_ptr = ctypes.c_int(index)
        label_size_ptr = ctypes.c_int(self.label_size)
        record_size_ptr = ctypes.c_int(self.record_size)

        result_ptr = self.binary_file_wrapper_cpp.get_label_native(
            self.file_path.encode('utf-8'), index_ptr, record_size_ptr, label_size_ptr)

        return result_ptr

    def get_label_cpp(self, index: int) -> int:
        """Get the label of the sample at the given index.

        Args:
            index (int): Index

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            int: Label for the sample
        """
        data = self.filesystem_wrapper.get(self.file_path)
        total_samples_ptr = ctypes.c_int(self.get_number_of_samples())
        index_ptr = ctypes.c_int(index)
        data_ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte))
        label_size_ptr = ctypes.c_int(self.label_size)
        record_size_ptr = ctypes.c_int(self.record_size)

        result_ptr = self.binary_file_wrapper_cpp.get_label(
            data_ptr, total_samples_ptr, index_ptr, record_size_ptr, label_size_ptr
        )

        result = result_ptr.value
        self.binary_file_wrapper_cpp.free_int(result_ptr)
        return result

    def get_all_labels(self) -> list[int]:
        """Returns a list of all labels of all samples in the file.

        Returns:
            list[int]: List of labels
        """
        if self._mode == 1:
            return self.get_all_labels_native_cpp()
        else:
            return self.get_all_labels_cpp()
        
    def get_all_labels_native_cpp(self) -> list[int]:
        """Returns a list of all labels of all samples in the file.

        Returns:
            list[int]: List of labels
        """
        number_of_samples = self.get_number_of_samples()
        num_samples_ptr = ctypes.c_int(number_of_samples)
        label_size_ptr = ctypes.c_int(self.label_size)
        record_size_ptr = ctypes.c_int(self.record_size)

        result_ptr = self.binary_file_wrapper_cpp.get_all_labels_native(
            self.file_path.encode('utf-8'), num_samples_ptr, record_size_ptr, label_size_ptr)
        
        labels = [result_ptr[i] for i in range(number_of_samples * self.label_size)]

        self.binary_file_wrapper_cpp.free(result_ptr)

        return labels

    def get_all_labels_cpp(self) -> list[int]:
        """Returns a list of all labels of all samples in the file.

        Returns:
            list[int]: List of labels
        """
        data = self.filesystem_wrapper.get(self.file_path)
        number_of_samples = self.get_number_of_samples()
        num_samples_ptr = ctypes.c_int(number_of_samples)
        data_ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte))
        label_size_ptr = ctypes.c_int(self.label_size)
        record_size_ptr = ctypes.c_int(self.record_size)

        result_ptr: IntVector = self.binary_file_wrapper_cpp.get_all_labels(
            data_ptr, num_samples_ptr, record_size_ptr, label_size_ptr
        )

        labels = [result_ptr[i].data for i in range(number_of_samples * self.label_size)]

        self.binary_file_wrapper_cpp.free(result_ptr)

        return labels

    def get_sample(self, index: int) -> bytes:
        """Get the sample at the given index.
        The indices are zero based.

        Args:
            index (int): Index

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        return self.get_samples_from_indices([index])[0]

    def get_samples(self, start: int, end: int) -> list[bytes]:
        """Get the samples at the given range from start (inclusive) to end (exclusive).
        The indices are zero based.

        Args:
            start (int): Start index
            end (int): End index

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        return self.get_samples_from_indices(list(range(start, end)))

    def get_samples_from_indices(self, indices: list) -> list[bytes]:
        """Get the samples at the given index list.
        The indices are zero based.

        Args:
            indices (list): List of indices of the required samples

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        self._validate_request_indices(indices)
        if self._mode == 1:
            return self.get_samples_from_indices_native_cpp(indices)
        else:
            return self.get_samples_from_indices_cpp(indices)
        
    def get_samples_from_indices_native_cpp(self, indices: list) -> list[bytes]:
        """Get the samples at the given index list.
        The indices are zero based.

        Args:
            indices (list): List of indices of the required samples

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        label_size_ptr = ctypes.c_int(self.label_size)
        record_size_ptr = ctypes.c_int(self.record_size)
        indices_ptr = IntVector((ctypes.c_int * len(indices))(*indices), len(indices))

        result_ptr = self.binary_file_wrapper_cpp.get_samples_from_indices_native(
            self.file_path.encode('utf-8'), indices_ptr, record_size_ptr, label_size_ptr)
        
        samples = [result_ptr[i] for i in range(len(indices) * (self.record_size - self.label_size))]

        self.binary_file_wrapper_cpp.free(result_ptr)

        return samples
    
    def get_samples_from_indices_cpp(self, indices: list) -> list[bytes]:
        """Get the samples at the given index list.
        The indices are zero based.

        Args:
            indices (list): List of indices of the required samples

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        data = self.filesystem_wrapper.get(self.file_path)
        data_ptr = ctypes.cast(data, ctypes.POINTER(ctypes.c_ubyte))
        label_size_ptr = ctypes.c_int(self.label_size)
        record_size_ptr = ctypes.c_int(self.record_size)
        indices_ptr = IntVector((ctypes.c_int * len(indices))(*indices), len(indices))

        result_ptr = self.binary_file_wrapper_cpp.get_samples_from_indices(
            data_ptr, indices_ptr, record_size_ptr, label_size_ptr
        )

        samples = [result_ptr[i] for i in range(len(indices) * (self.record_size - self.label_size))]

        self.binary_file_wrapper_cpp.free(result_ptr)

        return samples
