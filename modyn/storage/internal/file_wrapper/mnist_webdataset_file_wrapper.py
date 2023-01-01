import webdataset as wds

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class Mnist_webdatasetFileWrapper(AbstractFileWrapper):
    """MNIST Webdataset file wrapper."""

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_wrapper_type = FileWrapperType.MNIST_WEBDATASET

    def get_size(self) -> int:
        return wds.WebDataset(self.file_path).len()

    def get_samples(self, start: int, end: int) -> bytes:
        return wds.WebDataset(self.file_path).slice(start, end).decode("rgb").to_tuple("jpg;png;jpeg", "cls")

    def get_sample(self, index: int) -> bytes:
        return wds.WebDataset(self.file_path).slice(index, index + 1).decode("rgb").to_tuple("jpg;png;jpeg", "cls")
