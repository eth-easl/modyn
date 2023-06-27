import csv
from typing import Iterator, Optional

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


class CsvFileWrapper(AbstractFileWrapper):
    def __init__(self, file_path: str, file_wrapper_config: dict, filesystem_wrapper: AbstractFileSystemWrapper):
        super().__init__(file_path, file_wrapper_config, filesystem_wrapper)

        self.file_wrapper_type = FileWrapperType.CsvFileWrapper

        if "separator" in file_wrapper_config:
            self.separator = file_wrapper_config["separator"]
        else:
            self.separator = ","

        if "label_index" not in file_wrapper_config:
            raise ValueError("Please specify the index of the column that contains the label. ")
        if not isinstance(file_wrapper_config["label_index"], int) or file_wrapper_config["label_index"] < 0:
            raise ValueError("The label_index must be a positive integer.")
        self.label_index = file_wrapper_config["label_index"]

        # the first line might contain the header, which is useless and must not be returned.
        if "ignore_first_line" in file_wrapper_config:
            self.ignore_first_line = file_wrapper_config["ignore_first_line"]
        else:
            self.ignore_first_line = False

        if "encoding" in file_wrapper_config:
            self.encoding = file_wrapper_config["encoding"]
        else:
            self.encoding = "utf-8"

        # check that the file is actually a CSV
        self._validate_file_extension()

        # do not validate the content only if "validate_file_content" is explicitly set to False
        if ("validate_file_content" not in file_wrapper_config) or (
            "validate_file_content" in file_wrapper_config and file_wrapper_config["validate_file_content"]
        ):
            self._validate_file_content()

    def _validate_file_extension(self) -> None:
        """Validates the file extension as csv

        Raises:
            ValueError: File has wrong file extension
        """
        if not self.file_path.endswith(".csv"):
            raise ValueError("File has wrong file extension.")

    def _validate_file_content(self) -> None:
        """
        Performs the following checks:
        - specified label column is castable to integer
        - each row has the label_index_column
        - each row has the same width

        Raises a ValueError if a condition is not met
        """

        reader = self._get_csv_reader()

        number_of_columns = []

        for row in reader:
            number_of_columns.append(len(row))
            if not 0 <= self.label_index < len(row):
                raise ValueError("Label index outside row boundary")
            if not row[self.label_index].isnumeric():  # returns true iff all the characters are numbers
                raise ValueError("The label must be an integer")

        if len(set(number_of_columns)) != 1:
            raise ValueError(
                "Some rows have different width. " f"This is the number of columns row by row {number_of_columns}"
            )

    def get_sample(self, index: int) -> bytes:
        samples = self._filter_rows_samples([index])

        if len(samples) != 1:
            raise IndexError("Invalid index")

        return samples[0]

    def get_samples(self, start: int, end: int) -> list[bytes]:
        indices = list(range(start, end))
        return self.get_samples_from_indices(indices)

    def get_samples_from_indices(self, indices: list) -> list[bytes]:
        return self._filter_rows_samples(indices)

    def get_label(self, index: int) -> int:
        labels = self._filter_rows_labels([index])

        if len(labels) != 1:
            raise IndexError("Invalid index.")

        return labels[0]

    def get_all_labels(self) -> list[int]:
        reader = self._get_csv_reader()
        labels = [int(row[self.label_index]) for row in reader]
        return labels

    def get_number_of_samples(self) -> int:
        reader = self._get_csv_reader()
        return sum(1 for _ in reader)

    def _get_csv_reader(self) -> Iterator:
        """
        Receives the bytes from the file_system_wrapper and creates a csv.reader out of it.
        Returns:
            csv.reader
        """
        data_file = self.filesystem_wrapper.get(self.file_path)

        # Convert bytes content to a string
        data_file_str = data_file.decode(self.encoding)

        lines = data_file_str.split("\n")

        # Create a CSV reader
        reader = csv.reader(lines, delimiter=self.separator)

        # skip the header if required
        if self.ignore_first_line:
            next(reader)

        return reader

    def _filter_rows_samples(self, indices: list[int]) -> list[bytes]:
        """
        Filters the selected rows and removes the label column
        Args:
            indices: list of rows that must be kept

        Returns:
            list of byte-encoded rows

        """
        assert len(indices) == len(set(indices)), "An index is required more than once."
        reader = self._get_csv_reader()

        # Iterate over the rows and keep the selected ones
        filtered_rows: list[Optional[bytes]] = [None] * len(indices)
        for i, row in enumerate(reader):
            if i in indices:
                # Remove the label, convert the row to bytes and append to the list
                row_without_label = [col for j, col in enumerate(row) if j != self.label_index]
                # the row is transformed in a similar csv using the same separator and then transformed to bytes
                filtered_rows[indices.index(i)] = bytes(self.separator.join(row_without_label), self.encoding)

        if sum(1 for el in filtered_rows if el is None) != 0:
            raise IndexError("At least one index is invalid")

        # Here mypy complains that filtered_rows is a list of list[Optional[bytes]],
        # that can't happen given the above exception
        return filtered_rows  # type: ignore

    def _filter_rows_labels(self, indices: list[int]) -> list[int]:
        """
        Filters the selected rows and extracts the label column
        Args:
            indices: list of rows that must be kept

        Returns:
            list of labels

        """
        assert len(indices) == len(set(indices)), "An index is required more than once."
        reader = self._get_csv_reader()

        # Iterate over the rows and keep the selected ones
        filtered_rows: list[Optional[int]] = [None] * len(indices)
        for i, row in enumerate(reader):
            if i in indices:
                # labels are integer in modyn
                int_label = int(row[self.label_index])
                filtered_rows[indices.index(i)] = int_label

        if sum(1 for el in filtered_rows if el is None) != 0:
            raise IndexError("At least one index is invalid")

        # Here mypy complains that filtered_rows is a list of list[Optional[bytes]],
        # that can't happen given the above exception
        return filtered_rows  # type: ignore

    def delete_samples(self, indices: list) -> None:
        pass
