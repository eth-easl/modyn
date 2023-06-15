import os
from typing import Optional

import numpy as np
import torch
from modyn.common.trigger_sample.trigger_sample_storage import TriggerSampleStorage

LOCAL_STORAGE_FOLDER = ".tmp_offline_dataset"


class LocalDatasetHandler(TriggerSampleStorage):
    """
    Class that wraps TriggerSampleStorage to use it as a local storage for samples.
    This class is used to:
        - store (sample_id, score) for each selected sample in the dataset
        - retrieve these samples to train on
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        number_of_workers: int,
        maximum_keys_in_memory: Optional[int] = None,
    ) -> None:
        super().__init__(LOCAL_STORAGE_FOLDER)
        # files are numbered from 0. Each file has a size of number_of_samples_per_file
        self.current_file_index = 0
        self.current_sample_index = 0
        self.maximum_keys_in_memory = maximum_keys_in_memory  # None if we just use the class to read
        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.number_of_workers = number_of_workers
        if self.maximum_keys_in_memory is not None:
            # tuples are progressively accumulated in this ndarray and then dumped to file when the
            # desired file size is reached
            self.output_samples_list = np.empty(self.maximum_keys_in_memory, dtype=np.dtype("i8,f8"))

    def inform_samples(self, sample_ids: list, sample_weights: torch.Tensor) -> None:
        assert self.maximum_keys_in_memory is not None
        assert self.output_samples_list is not None

        samples_list = np.empty(len(sample_ids), dtype=np.dtype("i8,f8"))
        for i, _ in enumerate(sample_ids):
            samples_list[i] = (sample_ids[i], sample_weights[i])

        for element in samples_list:
            self.output_samples_list[self.current_sample_index] = element
            self.current_sample_index += 1

            if self.current_sample_index == self.maximum_keys_in_memory:
                self._samples_ready()

    def finalize(self) -> None:
        assert self.maximum_keys_in_memory is not None
        if self.current_sample_index > 0:
            self.output_samples_list = self.output_samples_list[: self.current_sample_index]  # remove empty elements
            self._samples_ready()

    def _samples_ready(self) -> None:
        assert self.maximum_keys_in_memory is not None
        assert self.output_samples_list is not None

        number_worker_samples = self.current_sample_index // self.number_of_workers

        for worker in range(self.number_of_workers):
            worker_samples = (
                self.output_samples_list[worker * number_worker_samples : (worker + 1) * number_worker_samples]
                if worker != self.number_of_workers - 1
                else self.output_samples_list[worker * number_worker_samples :]
            )
            path = self._get_file_name(
                self.pipeline_id, self.trigger_id, partition_id=self.current_file_index, worker_id=worker
            )
            self._write_file(path, worker_samples)

        self.current_sample_index = 0
        self.output_samples_list = np.empty(self.maximum_keys_in_memory, dtype=np.dtype("i8,f8"))
        self.current_file_index += 1

    def get_keys_and_weights(
        self,
        partition_id: int,
        worker_id: int,  # pylint: disable=unused-argument
    ) -> tuple[list[int], list[float]]:
        path = self._get_file_name(self.pipeline_id, self.trigger_id, partition_id, worker_id)
        file = path.parent / (path.name + ".npy")
        tuples_list = self._parse_file(file)

        keys = []
        weights = []
        for key, weight in tuples_list:
            keys.append(key)
            weights.append(weight)
        return keys, weights

    def clean_working_directory(self) -> None:
        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_pipeline_files = [
                file for file in os.listdir(LOCAL_STORAGE_FOLDER) if file.startswith(f"{self.pipeline_id}_")
            ]
            for file in this_pipeline_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))

    def clean_this_trigger_samples(self) -> None:
        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_trigger_files = [
                file
                for file in os.listdir(LOCAL_STORAGE_FOLDER)
                if file.startswith(f"{self.pipeline_id}_{self.trigger_id}_")
            ]
            for file in this_trigger_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))

    def get_number_of_partitions(self) -> int:
        # each file follows the structure {pipeline_id}_{trigger_id}_{partition_id}_{worker_id}

        # here we filter the files belonging to this pipeline and trigger
        this_trigger_partitions = [
            file
            for file in os.listdir(LOCAL_STORAGE_FOLDER)
            if file.startswith(f"{self.pipeline_id}_{self.trigger_id}_")
        ]

        # then we count how many partititions we have (not just len(this_trigger_partitions) since there could be
        # multiple workers for each partition
        return len(set(file.split("_")[2] for file in this_trigger_partitions))
