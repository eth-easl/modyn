from abc import ABC, abstractmethod

from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType

class AbstractFileWrapper(ABC):
    file_wrapper_type: FileWrapperType = None

    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def get_size(self) -> int:
        """
        Get the size of the file in number of samples.
        """
        raise NotImplementedError

    @abstractmethod
    def get_samples(self, start: int, end: int) -> bytes:
        """
        Get the samples from the file.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample(self, index: int) -> bytes:
        """
        Get the sample at the given index.
        """
        raise NotImplementedError
