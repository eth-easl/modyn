"""A file wrapper for files that contains only one sample and metadata."""

import pathlib

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class SingleSampleFileWrapper(AbstractFileWrapper):
    """A file wrapper for files that contains only one sample and metadata.

    For example, a file that contains only one image and metadata.
    The metadata is stored in a json file with the same name as the image file.
    """

    def __init__(self, file_path: str, file_wrapper_config: dict):
        """Init file wrapper.

        Args:
            file_path (str): File path
            file_wrapper_config (dict): File wrapper config
        """
        super().__init__(file_path, file_wrapper_config)
        self.file_wrapper_type = FileWrapperType.SingleSampleFileWrapper

    def get_number_of_samples(self) -> int:
        """Get the size of the file in number of samples.

        If the file has the correct file extension, it contains only one sample.

        Returns:
            int: Number of samples
        """
        if not self.file_path.endswith(self.file_wrapper_config["file_extension"]):
            return 0
        return 1

    def get_samples(self, start: int, end: int) -> bytes:
        """Get the samples from the file.

        Args:
            start (int): start index
            end (int): end index

        Raises:
            IndexError: If the start and end index are not 0 and 1

        Returns:
            bytes: Samples
        """
        if start != 0 or end != 1:
            raise IndexError("SingleSampleFileWrapper contains only one sample.")
        return self.get_sample(0)

    def get_sample(self, index: int) -> bytes:
        r"""Return the sample, the label and the length of the label as bytes.

        Format:
        sample + b'\n' + label + b'\n' + len(label).to_bytes(4, 'big')

        Args:
            index (int): Index

        Raises:
            ValueError: If the file has the wrong file extension
            IndexError: If the index is not 0

        Returns:
            bytes: Sample
        """
        if self.get_number_of_samples() == 0:
            raise ValueError("File has wrong file extension.")
        if index != 0:
            raise IndexError("SingleSampleFileWrapper contains only one sample.")
        with open(self.file_path, "rb") as file:
            label_path = pathlib.Path(self.file_path).with_suffix(self.file_wrapper_config["label_file_extension"])
            with open(label_path, "rb") as label_file:
                label = label_file.read()
                return file.read() + b"\n" + label + b"\n" + len(label).to_bytes(4, "big")

    def get_samples_from_indices(self, indices: list) -> bytes:
        """Get the samples from the file.

        Args:
            indices (list): Indices

        Raises:
            IndexError: If the indices are not valid

        Returns:
            bytes: Samples
        """
        if len(indices) != 1 or indices[0] != 0:
            raise IndexError("SingleSampleFileWrapper contains only one sample.")
        return self.get_sample(0)
