"""Webdataset file wrapper."""

import os
import pathlib
import pickle
import shutil
import uuid
from itertools import islice
from typing import Dict

import webdataset as wds
from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType
from modyn.storage.internal.filesystem_wrapper.abstract_filesystem_wrapper import AbstractFileSystemWrapper


class WebdatasetFileWrapper(AbstractFileWrapper):
    """Webdataset file wrapper.

    One file can contain multiple samples.

    This file wrapper is used for files that are in the webdataset file format.
    See here for more information about the webdataset file format:
    https://webdataset.github.io/webdataset/
    """

    tmp_dir: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent / "storage_tmp"

    def __init__(self, file_path: str, file_wrapper_config: dict, filesystem_wrapper: AbstractFileSystemWrapper):
        """Init webdataset file wrapper.

        Args:
            file_path (str): Path to file
            file_wrapper_config (dict): File wrapper config
        """
        super().__init__(file_path, file_wrapper_config, filesystem_wrapper)
        self.indeces_cache: Dict[str, str] = {}
        self.file_wrapper_type = FileWrapperType.WebdatasetFileWrapper

    def get_number_of_samples(self) -> int:
        """Get number of samples in file.

        This is a very slow operation. It is recommended to only use this method for testing purposes
        and for the initial loading of the dataset into the database.

        Returns:
            int: Number of samples in file
        """
        dataset = wds.WebDataset(self.file_path)
        length = 0
        for _ in dataset:
            length += 1
        return length

    def get_samples(self, start: int, end: int) -> bytes:
        """Get samples from start to end.

        Args:
            start (int): start index
            end (int): end index

        Returns:
            bytes: Pickled list of samples
        """
        return pickle.dumps(
            wds.WebDataset(self.file_path).slice(start, end).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json")
        )

    def get_sample(self, index: int) -> bytes:
        """Get sample from index.

        Args:
            index (int): Index of sample

        Returns:
            bytes: Pickled sample
        """
        return pickle.dumps(
            wds.WebDataset(self.file_path).slice(index, index + 1).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json")
        )

    def get_samples_from_indices(self, indices: list) -> bytes:
        """Get samples from indices.

        Args:
            indices (list): List of indices

        Returns:
            bytes: Pickled list of samples
        """
        indices.sort()

        if str(indices) in self.indeces_cache:
            file = self.indeces_cache[str(indices)]
            return pickle.dumps(wds.WebDataset(file).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json"))

        dataset = wds.WebDataset(self.file_path)

        file_name = uuid.uuid4().hex
        file = str(pathlib.Path(os.path.abspath(__file__)).parent / "storage_tmp" / f"{file_name}.tar")
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "wb") as tmp_file:
            with wds.TarWriter(tmp_file) as dst:
                index_start = indices[0]
                index_end = indices[0] - 1
                for i, index in enumerate(indices):
                    if index - index_end == 1:
                        index_end = index
                    else:
                        self.write_samples(dst, index_start, index_end, dataset)
                        index_start = index
                        index_end = index
                    if i == len(indices) - 1:
                        self.write_samples(dst, index_start, index_end, dataset)

        self.indeces_cache[str(indices)] = file

        return pickle.dumps(wds.WebDataset(file).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json"))

    def write_samples(self, dst: wds.TarWriter, index_start: int, index_end: int, dataset: wds.WebDataset) -> None:
        """Write samples to a tar file.

        Args:
            dst (wds.TarWriter): destination tar file
            index_start (int): index of the first sample to write
            index_end (int): index of the last sample to write
            dataset (wds.WebDataset): dataset to read the samples from
        """
        for sample in islice(dataset, index_start, index_end + 1):
            key = sample["__key__"]
            if "jpg" in sample:
                image = sample["jpg"]
                image_type = "jpg"
            elif "png" in sample:
                image = sample["png"]
                image_type = "png"
            elif "jpeg" in sample:
                image = sample["jpeg"]
                image_type = "jpeg"
            cls = sample["cls"]
            json = sample["json"]
            dst.write({"__key__": key, image_type: image, "cls": cls, "json": json})

    def __del__(self) -> None:
        """Delete the temporary files."""
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
