from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class SingleSampleFileWrapper(AbstractFileWrapper):
    """
    A file wrapper for files that contains only one sample.

    For example, a file that contains only one image.
    """

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_wrapper_type = FileWrapperType.SingleSampleFileWrapper

    def get_size(self) -> int:
        return 1

    def get_samples(self, start: int, end: int) -> bytes:
        if start != 0 or end != 1:
            raise IndexError('SingleSampleFileWrapper contains only one sample.')
        return self.get_sample(0)

    def get_sample(self, index: int) -> bytes:
        if index != 0:
            raise IndexError('SingleSampleFileWrapper contains only one sample.')
        with open(self.file_path, 'rb') as file:
            return file.read()

    def get_samples_from_indices(self, indices: list) -> bytes:
        if len(indices) != 1 or indices[0] != 0:
            raise IndexError('SingleSampleFileWrapper contains only one sample.')
        return self.get_sample(0)
