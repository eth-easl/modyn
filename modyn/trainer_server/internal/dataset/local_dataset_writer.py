import multiprocessing as mp
import os
import platform
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import torch
from modyn.common.trigger_sample.trigger_sample_storage import TriggerSampleStorage

LOCAL_STORAGE_FOLDER = ".tmp_offline_dataset"


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
    ) -> None:
        super().__init__(LOCAL_STORAGE_FOLDER)
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
        insertion_id: int,
    ) -> None:
        TriggerSampleStorage(
            trigger_sample_directory=LOCAL_STORAGE_FOLDER,
        ).save_trigger_sample(
            pipeline_id=pipeline_id,
            trigger_id=trigger_id,
            partition_id=partition_id,
            trigger_samples=training_samples,
            insertion_id=insertion_id,
        )

    def inform_samples(self, sample_ids: list, sample_weights: torch.Tensor) -> None:
        # map the two input lists to the desired format
        assert self.output_samples_list is not None
        samples_list = np.empty(len(sample_ids), dtype=np.dtype("i8,f8"))

        for i, _ in enumerate(sample_ids):
            samples_list[i] = (sample_ids[i], sample_weights[i])

        # add the input tuples to the output list.
        for element in samples_list:
            self.output_samples_list[self.current_sample_index] = element
            self.current_sample_index += 1

            # if the target is reached, store the samples
            if self.current_sample_index == self.maximum_keys_in_memory:
                self._samples_ready()

    def finalize(self) -> None:
        # store the remaining samples, if present
        if self.current_sample_index > 0:
            self.output_samples_list = self.output_samples_list[: self.current_sample_index]  # remove empty elements
            self._samples_ready()
            self._prepare_for_new_file()

    def _prepare_for_new_file(self) -> None:
        # reset counters and clean output list
        self.current_sample_index = 0
        self.output_samples_list = np.empty(self.maximum_keys_in_memory, dtype=np.dtype("i8,f8"))
        self.current_file_index += 1

    def _samples_ready(self) -> None:
        if (self._is_mac and self._is_test) or self._disable_mt:
            LocalDatasetWriter._store_triggersamples_impl(
                self.current_file_index,
                self.trigger_id,
                self.pipeline_id,
                np.array(self.output_samples_list, dtype=np.dtype("i8,f8")),
                0,
            )
            self._prepare_for_new_file()
            return

        number_worker_samples = self.current_sample_index // self.number_of_workers
        processes: list[mp.Process] = []

        with SharedMemoryManager() as smm:
            for i in range(self.number_of_workers):
                start_idx = i * number_worker_samples
                end_idx = (
                    start_idx + number_worker_samples
                    if i < self.number_of_workers - 1
                    else len(self.output_samples_list)
                )
                proc_samples = np.array(self.output_samples_list[start_idx:end_idx], dtype=np.dtype("i8,f8"))
                if len(proc_samples) > 0:
                    shm = smm.SharedMemory(proc_samples.nbytes)

                    shared_proc_samples: np.ndarray = np.ndarray(
                        proc_samples.shape, dtype=proc_samples.dtype, buffer=shm.buf
                    )
                    shared_proc_samples[:] = proc_samples  # This copies into the prepared numpy array
                    assert proc_samples.shape == shared_proc_samples.shape

                    proc = mp.Process(
                        target=LocalDatasetWriter._store_triggersamples_impl,
                        args=(self.current_file_index, self.trigger_id, self.pipeline_id, shared_proc_samples, i),
                    )
                    proc.start()
                    processes.append(proc)

            for proc in processes:
                proc.join()

        self._prepare_for_new_file()

    def clean_working_directory(self) -> None:
        # remove all the files belonging to this pipeline
        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_pipeline_files = list(
                filter(lambda file: file.startswith(f"{self.pipeline_id}_"), os.listdir(LOCAL_STORAGE_FOLDER))
            )

            for file in this_pipeline_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))

    def clean_this_trigger_samples(self) -> None:
        # remove all the files belonging to this pipeline and trigger

        if os.path.isdir(LOCAL_STORAGE_FOLDER):
            this_trigger_files = list(
                filter(
                    lambda file: file.startswith(f"{self.pipeline_id}_{self.trigger_id}_"),
                    os.listdir(LOCAL_STORAGE_FOLDER),
                )
            )

            for file in this_trigger_files:
                os.remove(os.path.join(LOCAL_STORAGE_FOLDER, file))

    def get_number_of_partitions(self) -> int:
        # each file follows the structure {pipeline_id}_{trigger_id}_{partition_id}_{worker_id}

        # here we filter the files belonging to this pipeline and trigger
        this_trigger_files = list(
            filter(
                lambda file: file.startswith(f"{self.pipeline_id}_{self.trigger_id}_"), os.listdir(LOCAL_STORAGE_FOLDER)
            )
        )

        # then we count how many partitions we have (not just len(this_trigger_partitions) since there could be
        # multiple workers for each partition
        return len(set(file.split("_")[2] for file in this_trigger_files))
