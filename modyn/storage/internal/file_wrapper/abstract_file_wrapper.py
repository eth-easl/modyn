from abc import ABC, abstractmethod

from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class AbstractFileWrapper(ABC):

    def __init__(self, file_path: str, file_wrapper_config: dict):
        self.file_wrapper_type: FileWrapperType = None
        self.file_path = file_path
        self.file_wrapper_config = file_wrapper_config

    @abstractmethod
    def get_number_of_samples(self) -> int:
        """
        Get the size of the file in number of samples.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_samples(self, start: int, end: int) -> bytes:
        """
        Get the samples from the file.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_sample(self, index: int) -> bytes:
        """
        Get the sample at the given index.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def get_samples_from_indices(self, indices: list) -> bytes:
        """
        Get the samples at the given indices.
        """
        raise NotImplementedError  # pragma: no cover
