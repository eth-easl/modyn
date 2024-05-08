from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np
import torch
from modyn.common.trigger_sample import TriggerSampleStorage


class LocalDatasetWriter(TriggerSampleStorage):
    """
    Class that wraps TriggerSampleStorage to use it as a local storage for samples.

    This Class is informed of samples and when the desired size is reached, it stores them in a file.
    Files can be stored using different workers in parallel or using just one worker.
    Furthermore, this class is used to map the usual format in the training loop (list of keys and list of weights) to
    the desired format of TriggerSampleStorage (list of tuples (key, weight)).
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        number_of_workers: int,
        maximum_keys_in_memory: int,
        offline_dataset_path: str,
    ) -> None:
        super().__init__(offline_dataset_path)
        # files are numbered from 0. Each file has a size of number_of_samples_per_file
        self.current_file_index = 0
        self.current_sample_index = 0
        self.maximum_keys_in_memory = maximum_keys_in_memory
        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        self.number_of_workers = number_of_workers
        self.output_samples_list = np.empty(self.maximum_keys_in_memory, dtype=np.dtype("i8,f8"))

        self._is_test = "PYTEST_CURRENT_TEST" in os.environ
        self._is_mac = platform.system() == "Darwin"
        self._disable_mt = self.number_of_workers <= 0

    @staticmethod
    def _store_triggersamples_impl(
        partition_id: int,
        trigger_id: int,
        pipeline_id: int,
        training_samples: np.ndarray,
        data_lengths: list,
        offline_dataset_path: str | Path,
    ) -> None:
        TriggerSampleStorage(trigger_sample_directory=offline_dataset_path).save_trigger_samples(
            pipeline_id=pipeline_id,
            trigger_id=trigger_id,
            partition_id=partition_id,
            trigger_samples=training_samples,
            data_lengths=data_lengths,
        )

    def inform_samples(self, sample_ids: list, sample_weights: torch.Tensor) -> None:
        # map the two input lists to the desired format
        assert self.output_samples_list is not None

        # add the input tuples to the output list.
        for sample_id, sample_weight in zip(sample_ids, sample_weights):
            self.output_samples_list[self.current_sample_index] = (sample_id, sample_weight)
            self.current_sample_index += 1

            # if the target is reached, store the samples
            if self.current_sample_index == self.maximum_keys_in_memory:
                self._samples_ready()

    def finalize(self) -> None:
        # store the remaining samples, if present
        if self.current_sample_index > 0:
            self.output_samples_list = self.output_samples_list[: self.current_sample_index]  # remove empty elements
            self._samples_ready()

    def _prepare_for_new_file(self) -> None:
        # reset counters and clean output list
        self.current_sample_index = 0
        self.output_samples_list = np.empty(self.maximum_keys_in_memory, dtype=np.dtype("i8,f8"))
        self.current_file_index += 1

    def _samples_ready(self) -> None:
        # TODO(#276) Unify AbstractSelection Strategy and LocalDatasetWriter
        if (self._is_mac and self._is_test) or self._disable_mt:
            LocalDatasetWriter._store_triggersamples_impl(
                self.current_file_index,
                self.trigger_id,
                self.pipeline_id,
                np.array(self.output_samples_list, dtype=np.dtype("i8,f8")),
                [len(self.output_samples_list)],
                self.trigger_sample_directory,
            )
            self._prepare_for_new_file()
            return

        samples_per_proc = self.current_sample_index // self.number_of_workers

        data_lengths = []
        if samples_per_proc > 0:
            data_lengths = [samples_per_proc] * (self.number_of_workers - 1)

        if sum(data_lengths) < len(self.output_samples_list):
            data_lengths.append(len(self.output_samples_list) - sum(data_lengths))

        LocalDatasetWriter._store_triggersamples_impl(
            self.current_file_index,
            self.trigger_id,
            self.pipeline_id,
            np.array(self.output_samples_list, dtype=np.dtype("i8,f8")),
            data_lengths,
            self.trigger_sample_directory,
        )

        self._prepare_for_new_file()

    def clean_this_trigger_samples(self) -> None:
        self.clean_trigger_data(self.pipeline_id, self.trigger_id)

    def get_number_of_partitions(self) -> int:
        return self.get_trigger_num_data_partitions(self.pipeline_id, self.trigger_id)
