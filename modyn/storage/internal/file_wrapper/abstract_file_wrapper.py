"""Base class for all file wrappers."""

from abc import ABC, abstractmethod
from typing import Optional

from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


class AbstractFileWrapper(ABC):
    """Base class for all file wrappers."""

    def __init__(self, file_path: str, file_wrapper_config: dict, filesystem_wrapper: AbstractFileSystemWrapper):
        """Init file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
        """
        self.file_wrapper_type: FileWrapperType = None
        self.file_path = file_path
        self.file_wrapper_config = file_wrapper_config
        self.filesystem_wrapper = filesystem_wrapper

    @abstractmethod
    def get_number_of_samples(self) -> int:
        """Get the size of the file in number of samples.

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            int: Number of samples
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_samples(self, start: int, end: int) -> list[bytes]:
        """Get the samples from the file.

        Args:
            start (int): Start index
            end (int): End index

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bytes: Samples
        """
        raise NotImplementedError  # pragma: no cover

    def get_label(self, index: int) -> Optional[int]:
        """Get the label at the given index.

        Args:
            index (int): Index

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            int: Label if exists, else None
        """
        raise NotImplementedError  # pragma: no cover

    def get_all_labels(self) -> list[Optional[int]]:
        """Returns a list of all labels of all samples in the file.

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            list[Optional[int]]: List of labels
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_sample(self, index: int) -> bytes:
        """Get the sample at the given index.

        Args:
            index (int): Index

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bytes: Sample
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_samples_from_indices(self, indices: list) -> list[bytes]:
        """Get the samples at the given indices.

        Args:
            indices (list): List of indices

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bytes: Samples
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def delete_samples(self, indices: list) -> None:
        """Delete the samples at the given indices.

        Args:
            indices (list): List of indices

        Raises:
            NotImplementedError: If the method is not implemented
        """
        raise NotImplementedError
