import os
import shutil
from typing import Optional

import numpy as np
from modyn.common.trigger_sample.trigger_sample_storage import TriggerSampleStorage

LOCAL_STORAGE_FOLDER = ".tmp_offline_dataset"


class LocalDatasetHandler(TriggerSampleStorage):
    """
    Class that wraps TriggerSampleStorage to use it as a local storage for samples.
    This class is used to:
        - store (sample_id, score) for each selected sample in the dataset
        - retrieve these samples to train on
    """

    def __init__(self, pipeline_id: int, number_of_samples_per_file: Optional[int] = None) -> None:
        super().__init__(LOCAL_STORAGE_FOLDER)
        # files are numbered from 0. Each file has a size of number_of_samples_per_file
        self.current_file_index = 0
        self.current_sample_index = 0
        self.file_size = number_of_samples_per_file  # None if we just use the class to read
        self.pipeline_id = pipeline_id
        if self.file_size is not None:
            # tuples are progressively accumulated in this ndarray and then dumped to file when the
            # desired file size is reached
            self.output_samples_list = np.empty(self.file_size, dtype=np.dtype("i8,f8"))

    def inform_samples(self, input_samples_list: np.ndarray) -> None:
        assert self.file_size is not None
        assert self.output_samples_list is not None
        for element in input_samples_list:
            self.output_samples_list[self.current_sample_index] = element
            self.current_sample_index += 1

            if self.current_sample_index == self.file_size:
                self._samples_ready()

    def store_last_samples(self) -> None:
        assert self.file_size is not None
        if self.current_sample_index > 0:
            self.output_samples_list = self.output_samples_list[: self.current_sample_index]  # remove empty elements
            self._samples_ready()

    def _samples_ready(self) -> None:
        assert self.file_size is not None
        assert self.output_samples_list is not None
        self.save_trigger_sample(self.pipeline_id, 0, self.current_file_index, self.output_samples_list, -1)
        self.current_sample_index = 0
        self.output_samples_list = np.empty(self.file_size, dtype=np.dtype("i8,f8"))
        self.current_file_index += 1

    def get_keys_and_weights(
        self, worker_id: int, partition_id: int  # pylint: disable=unused-argument
    ) -> tuple[list[int], list[float]]:
        tuples_list = self.get_trigger_samples(self.pipeline_id, 0, partition_id)
        keys = []
        weights = []
        for key, weight in tuples_list:
            keys.append(key)
            weights.append(weight)
        return keys, weights

    def clean_working_directory(self) -> None:
        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            shutil.rmtree(LOCAL_STORAGE_FOLDER)
