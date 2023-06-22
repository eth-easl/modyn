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
            self.separator = ";"

        if "label_index" not in file_wrapper_config:
            raise ValueError(
                "Please specify the index of the column that contains the label. "
                "Use None if no column contains the label"
            )
        self.label_index = file_wrapper_config["label_index"]

        if "ignore_first_line" in file_wrapper_config:
            self.ignore_first_line = file_wrapper_config["ignore_first_line"]
        else:
            self.ignore_first_line = False

        if "encoding" in file_wrapper_config:
            self.encoding = file_wrapper_config["encoding"]
        else:
            self.encoding = "utf-8"

        self._validate_file_extension()

    def _validate_file_extension(self) -> None:
        """Validates the file extension as csv

        Raises:
            ValueError: File has wrong file extension
        """
        if not self.file_path.endswith(".csv"):
            raise ValueError("File has wrong file extension.")

    def get_sample(self, index: int) -> bytes:
        samples = self._filter_rows_samples([index])

        if len(samples) != 1:
            raise IndexError("Invalid index")

        return samples[0]

    def get_samples(self, start: int, end: int) -> list[bytes]:
        indices = list(range(start, end))
        return self.get_samples_from_indices(indices)

    def get_samples_from_indices(self, indices: list) -> list[bytes]:
        samples = self._filter_rows_samples(indices)

        if len(samples) != len(indices):
            raise IndexError("At least one index is invalid.")

        return samples

    def get_label(self, index: int) -> Optional[int]:
        if self.label_index is None:
            return None

        labels = self._filter_rows_labels([index])

        if len(labels) != 1:
            raise IndexError("Invalid index.")

        return labels[0]

    def get_all_labels(self) -> list[Optional[int]]:
        reader = self._get_csv_reader()

        if self.label_index is None:
            return [None] * self.get_number_of_samples()

        labels = []
        for row in reader:
            only_label = row[self.label_index]
            labels.append(only_label)

        return labels

    def get_number_of_samples(self) -> int:
        reader = self._get_csv_reader()

        return sum(1 for _ in reader)

    def _get_csv_reader(self) -> Iterator:
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
        reader = self._get_csv_reader()

        # Iterate over the rows and keep the selected ones
        filtered_rows = []
        for i, row in enumerate(reader):
            if i in indices:
                # Remove the label, convert the row to bytes and append to the list
                row_without_label = [col for j, col in enumerate(row) if j != self.label_index]
                filtered_rows.append(bytes(self.separator.join(row_without_label), self.encoding))

        return filtered_rows

    def _filter_rows_labels(self, indices: list[int]) -> list[int]:
        reader = self._get_csv_reader()

        # Iterate over the rows and keep the selected ones
        filtered_rows = []
        for i, row in enumerate(reader):
            if i in indices:
                only_label = row[self.label_index]
                filtered_rows.append(only_label)

        return filtered_rows

    def delete_samples(self, indices: list) -> None:
        return
