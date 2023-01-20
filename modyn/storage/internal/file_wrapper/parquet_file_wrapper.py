"""Parquet file wrapper."""

import pyarrow.fs as fs
import pyarrow.parquet as pq

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper
from modyn.storage.internal.filesystem_wrapper.filesystem_wrapper_type import FilesystemWrapperType


class ParquetFileWrapper(AbstractFileWrapper):
    """Webdataset file wrapper.

    One file can contain multiple samples.

    This file wrapper is used for files that are in the parquet file format.
    See here for more information about the parquet file format:
    https://parquet.apache.org/docs/
    """

    def __init__(self, file_path: str, file_wrapper_config: dict, filesystem_wrapper: AbstractFileSystemWrapper):
        """Init parquet file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
            filesystem_wrapper (AbstractFileSystemWrapper): File system wrapper to abstract storage of the file
        """
        super().__init__(file_path, file_wrapper_config, filesystem_wrapper)
        self.file_wrapper_type = FileWrapperType.WebdatasetFileWrapper
        self._setup_pq_filesystem(filesystem_wrapper)

    def _setup_pq_filesystem(self, filesystem_wrapper: AbstractFileSystemWrapper):
        """
        Setup the appropriate Parquet Filesystem for accessing remote files

        Args:
            filesystem_wrapper (AbstractFileSystemWrapper): File system wrapper to abstract storage of the file
        """
        fs_wrapper_type = filesystem_wrapper.filesystem_wrapper_type
        if(fs_wrapper_type == FilesystemWrapperType.LocalFilesystemWrapper):
            self.pq_filesystem = fs.LocalFileSystem()
        else:
            raise NotImplementedError("Unsupport filesystem type for Parquet files")

    def _get_data(self):
        """
        Reads the table from the initialized parquet file path from the corresponding filesystem

        Returns:
            pyarrow.lib.Table: Parquet Table object read from the file
        """
        return pq.read_table(self.file_path, filesystem=self.pq_filesystem)

    def get_number_of_samples(self) -> int:
        """Get number of samples in file.

        Returns:
            int: Number of samples in file
        """
        data_table = self._get_data()
        return data_table.num_rows

    def get_sample(self, index: int):
        """Get the sample at the given index.

        Args:
            index (int): Index

        Returns:
            bytes: Sample
        """
        return self.get_samples(index, index+1)

    def get_samples(self, start: int, end: int):
        """Get the samples at the given range from start (inclusive) to end (exclusive).

        Args:
            start (int): Start index
            end (int): End index

        Returns:
            bytes: Sample
        """
        data = self._get_data().to_pandas()
        data_records = data.iloc[start:end].to_records()
        sample_bytes = data_records.tobytes()
        return sample_bytes

    def get_samples_from_indices(self, indices: list):
        """Get the samples at the given index list.

        Args:
            indices (list): List of indices of the required samples

        Returns:
            bytes: Sample
        """
        data = self._get_data().to_pandas()
        data_records = data.iloc[indices].to_records()
        sample_bytes = data_records[indices].tobytes()
        return sample_bytes
