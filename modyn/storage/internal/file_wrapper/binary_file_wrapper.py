"""Binary file wrapper."""

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


class BinaryFileWrapper(AbstractFileWrapper):
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
        self.file_wrapper_type = FileWrapperType.BinaryFileWrapper
        self.byteorder = file_wrapper_config["byteorder"]

        self.record_size = file_wrapper_config["record_size"]
        self.label_size = file_wrapper_config["label_size"]
        if self.record_size - self.label_size < 1:
            raise ValueError("Each record must have at least 1 byte of data other than the label.")

        self._validate_file_extension()
        self.file_size = self.filesystem_wrapper.get_size(self.file_path)
        if self.file_size % self.record_size != 0:
            raise ValueError("File does not contain exact number of records of size " + str(self.record_size))

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
        invalid_indices = any((idx < 0 or idx > (total_samples - 1)) for idx in indices)
        if invalid_indices:
            raise IndexError("Indices are out of range. Indices should be between 0 and " + str(total_samples))

    def get_number_of_samples(self) -> int:
        """Get number of samples in file.

        Returns:
            int: Number of samples in file
        """
        return self.file_size / self.record_size

    def get_label(self, index: int) -> int:
        """Get the label of the sample at the given index.

        Args:
            index (int): Index

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            int: Label for the sample
        """
        data = self.filesystem_wrapper.get(self.file_path)

        total_samples = len(data) / self.record_size
        self._validate_request_indices(total_samples, [index])

        record_start = index * self.record_size
        lable_bytes = data[record_start: record_start + self.label_size]
        return int.from_bytes(lable_bytes, byteorder=self.byteorder)

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
        data = self.filesystem_wrapper.get(self.file_path)

        total_samples = len(data) / self.record_size
        self._validate_request_indices(total_samples, indices)

        samples = [data[(idx * self.record_size) + self.label_size: (idx + 1) * self.record_size] for idx in indices]
        return samples
