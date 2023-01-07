"""Base class for all file wrappers."""

from abc import ABC, abstractmethod

from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class AbstractFileWrapper(ABC):
    """Base class for all file wrappers."""

    def __init__(self, file_path: str, file_wrapper_config: dict):
        """Init file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
        """
        self.file_wrapper_type: FileWrapperType = None
        self.file_path = file_path
        self.file_wrapper_config = file_wrapper_config

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
    def get_samples(self, start: int, end: int) -> bytes:
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
    def get_samples_from_indices(self, indices: list) -> bytes:
        """Get the samples at the given indices.

        Args:
            indices (list): List of indices

        Raises:
            NotImplementedError: If the method is not implemented

        Returns:
            bytes: Samples
        """
        raise NotImplementedError  # pragma: no cover
