"""CSV file wrapper."""

import pandas as pd
from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


class CSVFileWrapper(AbstractFileWrapper):
    """CSV file wrapper.

    CSV files store samples data in a row-oriented format. CSV file can have a header. The header line index can
    be specified in the file_wrapper_config while creating an instance. The file wrapper is able to read samples by
    index, and value by feature name / col index and row index.
    """

    def __init__(
        self,
        file_path: str,
        file_wrapper_config: dict,
        filesystem_wrapper: AbstractFileSystemWrapper,
    ):
        """Init CSV file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
            filesystem_wrapper (AbstractFileSystemWrapper): File system wrapper to abstract storage of the file

        Raises:
            ValueError: If the file has the wrong file extension
            ValueError: If the file does not contain an exact number of samples of given size
        """
        super().__init__(file_path, file_wrapper_config, filesystem_wrapper)
        self.file_wrapper_type = FileWrapperType.CSVFileWrapper
        self.ncols = file_wrapper_config["number_cols"]
        if self.ncols is None or self.ncols < 1:
            raise ValueError("Each csv should have at least one column.")

        self.delim = file_wrapper_config["delimiter"]
        if self.delim is None or self.delim == "":
            self.delim = ","

        self.extension = file_wrapper_config["file_extension"]
        if self.extension is None or self.extension == "":
            self.extension = ".csv"

        self.encoding = file_wrapper_config["encoding"]
        if self.encoding is None or self.encoding == "":
            self.extension = "utf-8"

        self.header = file_wrapper_config["header"]
        if self.header is None:
            self.header = None

        self._validate_file_extension()

    def _validate_file_extension(self) -> None:
        """Validates the file extension as bin

        Raises:
            ValueError: File has wrong file extension
        """
        if not self.file_path.endswith(self.extension):
            raise ValueError("File has wrong file extension.")

    def get_number_of_samples(self) -> int:
        """Get number of samples in file.

        Returns:
            int: Number of samples in file
        """
        i = 0
        with open(self.file_path, "r", encoding=self.encoding) as file_pointer:
            for _ in file_pointer:
                i += 1
        return i - (1 if self.header is not None else 0)

    def get_label(self, index: int) -> bytes:
        """Get the label of the sample at the given index.

        Args:
            index (int): Index

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            object: Label for the sample
        """
        data_frame = pd.read_csv(self.file_path, sep=self.delim, header=self.header, encoding=self.encoding)
        return bytes(data_frame.columns[index], self.encoding)

    def get_all_labels(self) -> list[bytes]:
        """Returns a list of all labels of all samples in the file.

        Returns:
            list: List of labels
        """
        data_frame = pd.read_csv(self.file_path, sep=self.delim, header=self.header, encoding=self.encoding)
        return [bytes(col, self.encoding) for col in data_frame.columns.tolist()]

    def get_sample(self, index: int) -> list[bytes]:
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

    def get_samples(self, start: int, end: int) -> list[list[bytes]]:
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
        return self.get_samples_from_indices([*range(start, end)])

    def get_samples_from_indices(self, indices: list) -> list[list[bytes]]:
        """Get the samples at the given index list.
        The indices are zero based.

        Args:
            indices (list): List of indices of the required samples

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        data_frame = pd.read_csv(self.file_path, sep=self.delim, header=self.header, encoding=self.encoding)
        samples = data_frame.iloc[indices].values.tolist()
        res = [[bytes(val, self.encoding) for val in row] for row in samples]
        return res

    def get_values_from_features_indices(self, features_indices: list) -> list[bytes]:
        """Get the samples at the given list of feature name (str) and index (int).
        The indices are zero based.

        Args:
            features_indices (list): List of feature names (str) and indices (int) of the required samples

        Raises:
            IndexError: If the index is out of bounds

        Returns:
            bytes: Sample
        """
        data_frame = pd.read_csv(self.file_path, sep=self.delim, header=self.header, encoding=self.encoding)
        samples = [bytes(data_frame.get_value(index, feature)) for feature, index in features_indices]
        return samples

    def delete_samples(self, indices: list) -> None:
        """Delete the samples at the given index list.
        The indices are zero based.

        We do not support deleting samples from CSV files.
        We can only delete the entire file which is done when every sample is deleted.
        This is done to avoid the overhead of updating the file after every deletion.

        See remove_empty_files in the storage grpc servicer for more details.

        Args:
            indices (list): List of indices of the samples to delete
        """
        return
