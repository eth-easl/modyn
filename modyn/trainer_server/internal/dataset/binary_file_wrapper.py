"""Binary file wrapper."""
import os


class BinaryFileWrapper:
    """Binary file wrapper.

    Binary files store raw sample data in a row-oriented format. One file can contain multiple samples.
    This wrapper requires that each samples should start with the label followed by its set of features.
    Each sample should also have a fixed overall width (in bytes) and a fixed width for the label,
    both of which should be provided in the config. The file wrapper is able to read samples by
    offsetting the required number of bytes.
    """

    def __init__(self, file_path: str, byteorder: str, record_size: int, label_size: int):
        """Init binary file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
            filesystem_wrapper (AbstractFileSystemWrapper): File system wrapper to abstract storage of the file

        Raises:
            ValueError: If the file has the wrong file extension
            ValueError: If the file does not contain an exact number of samples of given size
        """
        self.byteorder = byteorder
        self.file_path = file_path

        self.record_size = record_size
        self.label_size = label_size
        if self.record_size - self.label_size < 1:
            raise ValueError("Each record must have at least 1 byte of data other than the label.")

        self.file_size = os.path.getsize(self.file_path)

        if self.file_size % self.record_size != 0:
            raise ValueError("File does not contain exact number of records of size " + str(self.record_size))

    def get_number_of_samples(self) -> int:
        """Get number of samples in file.

        Returns:
            int: Number of samples in file
        """
        return int(self.file_size / self.record_size)

    def get_all_labels(self) -> list[int]:
        with open(self.file_path, "rb") as file:
            data = file.read()

        num_samples = self.get_number_of_samples()
        labels = [
            int.from_bytes(
                data[(idx * self.record_size) : (idx * self.record_size) + self.label_size], byteorder=self.byteorder
            )
            for idx in range(num_samples)
        ]
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
        return self.get_samples_from_indices(list(range(start, end)))

    def get_samples_from_indices(self, indices: list) -> list[bytes]:
        with open(self.file_path, "rb") as file:
            data = file.read()

        samples = [data[(idx * self.record_size) + self.label_size : (idx + 1) * self.record_size] for idx in indices]
        return samples
