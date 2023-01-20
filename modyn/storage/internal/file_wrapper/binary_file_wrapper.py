"""Binary file wrapper."""

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


class BinaryFileWrapper(AbstractFileWrapper):
    """Binary file wrapper.

    One file can contain multiple samples. Each samples should have a fixed overall 
    width (in bytes) which should be provided in the config. The file wrapper is able
    to read samples by offsetting the required number of bytes.
    """

    def __init__(self, file_path: str, file_wrapper_config: dict, filesystem_wrapper: AbstractFileSystemWrapper):
        """Init binary file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
            filesystem_wrapper (AbstractFileSystemWrapper): File system wrapper to abstract storage of the file
        """
        super().__init__(file_path, file_wrapper_config, filesystem_wrapper)
        self.file_wrapper_type = FileWrapperType.BinaryFileWrapper
        self.record_size = file_wrapper_config["record_size"]
        self.label_offset = file_wrapper_config["label_offset"]
        self.label_size = file_wrapper_config["label_size"]

    def _validate_file_extension(self):
        """Validates the file extension as bin

        Raises:
            ValueError: File has wrong file extension
        """
        if not(self.file_path.endswith(".bin")):
            raise ValueError("File has wrong file extension.")

    def get_number_of_samples(self) -> int:
        """Get number of samples in file.

        Returns:
            int: Number of samples in file
        """
        self._validate_file_extension()

        file_size = self.filesystem_wrapper.get_size(self.file_path)
        return file_size / self.record_size

    def get_sample(self, index: int):
        """Get the sample at the given index.
        The indices are zero based.

        Args:
            index (int): Index

        Raises:
            ValueError: If the file has the wrong file extension
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        return self.get_samples(index, index+1)

    def get_label(self, index: int) -> int:
        """Get the label of the sample at the given index.

        Args:
            index (int): Index

        Raises:
            ValueError: If the file has the wrong file extension
            IndexError: If the index is out of bounds

        Returns:
            int: Label for the sample
        """
        sample = self.get_sample(index)
        lable_bytes = sample[self.label_offset: self.label_offset + self.label_size]
        return int.from_bytes(lable_bytes, byteorder="big")


    def get_samples(self, start: int, end: int):
        """Get the samples at the given range from start (inclusive) to end (exclusive).
        The indices are zero based.

        Args:
            start (int): Start index
            end (int): End index

        Raises:
            ValueError: If the file has the wrong file extension
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        self._validate_file_extension()
        data = bytearray(self.filesystem_wrapper.get(self.file_path))

        total_samples = len(data) / self.record_size
        invalid_start =  (start > (total_samples - 1) or start < 0 )
        invalid_end = (end < 1 or end > total_samples)
        if(invalid_start or invalid_end):
            raise IndexError("Indices are out of range.")

        start_offset = start * self.record_size
        end_offset = end * self.record_size
        return data[start_offset: end_offset]

    def get_samples_from_indices(self, indices: list):
        """Get the samples at the given index list.
        The indices are zero based.

        Args:
            indices (list): List of indices of the required samples

        Raises:
            ValueError: If the file has the wrong file extension
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        self._validate_file_extension()
        data = bytearray(self.filesystem_wrapper.get(self.file_path))

        total_samples = len(data) / self.record_size
        invalid_indices = any((idx < 0 or idx > (total_samples - 1)) for idx in indices)
        if(invalid_indices):
            raise IndexError("Indices are out of range.")

        samples = bytearray()
        for idx in indices:
            sample = data[idx*self.record_size: (idx+1)*self.record_size]
            samples += sample
        return samples
        
