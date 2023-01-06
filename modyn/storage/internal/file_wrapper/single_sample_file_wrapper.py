import pathlib

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class SingleSampleFileWrapper(AbstractFileWrapper):
    """
    A file wrapper for files that contains only one sample.

    For example, a file that contains only one image.
    """

    def __init__(self, file_path: str, file_wrapper_config: dict):
        super().__init__(file_path, file_wrapper_config)
        self.file_wrapper_type = FileWrapperType.SingleSampleFileWrapper

    def get_number_of_samples(self) -> int:
        """
        If the file has the correct file extension, it contains only one sample.
        """
        if not self.file_path.endswith(self.file_wrapper_config['file_extension']):
            return 0
        return 1

    def get_samples(self, start: int, end: int) -> bytes:
        if start != 0 or end != 1:
            raise IndexError('SingleSampleFileWrapper contains only one sample.')
        return self.get_sample(0)

    def get_sample(self, index: int) -> bytes:
        """
        Returns the sample, the label and the length of the label as bytes.

        Format:
        sample + b'\n' + label + b'\n' + len(label).to_bytes(4, 'big')
        """
        if self.get_number_of_samples() == 0:
            raise ValueError('File has wrong file extension.')
        if index != 0:
            raise IndexError('SingleSampleFileWrapper contains only one sample.')
        with open(self.file_path, 'rb') as file:
            label_path = pathlib.Path(self.file_path).with_suffix(self.file_wrapper_config['label_file_extension'])
            with open(label_path, 'rb') as label_file:
                label = label_file.read()
                return file.read() + b'\n' + label + b'\n' + len(label).to_bytes(4, 'big')

    def get_samples_from_indices(self, indices: list) -> bytes:
        if len(indices) != 1 or indices[0] != 0:
            raise IndexError('SingleSampleFileWrapper contains only one sample.')
        return self.get_sample(0)
