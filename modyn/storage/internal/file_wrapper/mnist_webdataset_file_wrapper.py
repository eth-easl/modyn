import pickle
import os
from itertools import islice
import uuid
from typing import Dict

import webdataset as wds

from modyn.storage.internal.file_wrapper.abstract_file_wrapper import AbstractFileWrapper
from modyn.storage.internal.file_wrapper.file_wrapper_type import FileWrapperType


class MNISTWebdatasetFileWrapper(AbstractFileWrapper):
    """MNIST Webdataset file wrapper."""
    indeces_cache: Dict[str, str] = {}

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_wrapper_type = FileWrapperType.MNISTWebdatasetFileWrapper

    def get_size(self) -> int:
        """
        This is a very slow operation. It is recommended to only use this method for testing purposes
        and for the initial loading of the dataset into the database.
        """
        dataset = wds.WebDataset(self.file_path)
        length = 0
        for _ in dataset:
            length += 1
        return length

    def get_samples(self, start: int, end: int) -> bytes:
        return pickle.dumps(wds.WebDataset(self.file_path)
                            .slice(start, end).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json"))

    def get_sample(self, index: int) -> bytes:
        return pickle.dumps(wds.WebDataset(self.file_path)
                            .slice(index, index + 1).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json"))

    def get_samples_from_indices(self, indices: list) -> bytes:

        indices.sort()

        if str(indices) in self.indeces_cache:
            file_path = self.indeces_cache[str(indices)]
            return pickle.dumps(wds.WebDataset(file_path).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json"))

        dataset = wds.WebDataset(self.file_path)

        file_name = uuid.uuid4().hex
        file_path = os.getcwd() + os.path.sep + os.path.join('storage_tmp', 'modyn', 'mnist', f'{file_name}.tar')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as tmp_file:
            with wds.TarWriter(tmp_file) as dst:
                index_start = indices[0]
                index_end = indices[0] - 1
                for i, index in enumerate(indices):
                    print(f"index: {index}")
                    if index - index_end == 1:
                        index_end = index
                    else:
                        self.write_samples(dst, index_start, index_end, dataset)
                        index_start = index
                        index_end = index
                    if i == len(indices) - 1:
                        self.write_samples(dst, index_start, index_end, dataset)

        self.indeces_cache[str(indices)] = file_path

        return pickle.dumps(wds.WebDataset(file_path).decode("rgb").to_tuple("jpg;png;jpeg", "cls", "json"))

    def write_samples(self, dst: wds.TarWriter, index_start: int, index_end: int, dataset: wds.WebDataset) -> None:
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
