import os.path
import shutil
from typing import Any

import numpy as np
import torch
from modyn.common.trigger_sample import TriggerSampleStorage
from typing_extensions import Self

TEMPORARY_LOCAL_STORAGE_PATH = ".tmp_scores"  # should we provide it through the config?


class SampleThenBatchHandler:
    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        downsampled_batch_ratio: int,
        maximum_keys_in_memory: int,
    ) -> None:
        # used to temporarily store the computed scores
        self.scores_storage = TriggerSampleStorage(TEMPORARY_LOCAL_STORAGE_PATH)

        self.pipeline_id = pipeline_id
        self.trigger_id = trigger_id
        assert batch_size > 0
        self.batch_size = batch_size
        assert maximum_keys_in_memory > 0
        self.maximum_keys_in_memory = maximum_keys_in_memory
        assert 0 < downsampled_batch_ratio < 100
        self.downsampled_batch_ratio = downsampled_batch_ratio

        # below a list of arguments that are used throughout the function but not yet available
        self.to_be_stored: list[tuple[int, float]] = []
        self.current_file_index = 0
        self.current_scores_sum = 0.0
        self.file_total_scores: list[float] = []
        self.grouped_samples_per_file: list[int] = []
        self.number_of_samples_per_file: list[int] = []
        self.number_of_samples_seen = 0
        self.normalizer = 0.0

    def __enter__(self) -> Self:
        """
        Used in the context manager. Checks that everything is available and resets the counters
        Returns: Self

        """
        assert self.batch_size > 0
        assert self.pipeline_id >= 0
        assert 0 < self.downsampled_batch_ratio < 100
        self.to_be_stored = []
        self.current_file_index = 0
        self.current_scores_sum = 0.0
        self.file_total_scores = []
        self.number_of_samples_per_file = []
        self.normalizer = 0.0
        self._clean_working_directory()
        return self

    def accumulate(self, sample_ids: list, scores: torch.Tensor) -> None:
        """
        Function to accumulate ids and scores. These values are temporarily stored using TriggerSampleStorage
        Args:
            sample_ids: list of sample ids
            scores: tensor of scores
            batch_number: index of the batch
        """
        assert len(sample_ids) == scores.shape[0]

        self.normalizer += scores.sum().item()
        for sample_id, score in zip(sample_ids, scores):
            self.to_be_stored.append((sample_id, score))
            self.current_scores_sum += score.item()
            if len(self.to_be_stored) == self.maximum_keys_in_memory:
                self._store_values()

    def __exit__(self, *exc: Any) -> None:
        """
        Used in the context manager.
        When all datapoints have been accumulated, the number of points per file are sampled.
        """
        assert exc[0] is None, f"Something went wrong: {exc}"

        # store the last samples
        if len(self.to_be_stored) > 0:
            self._store_values()

        # aggregate per-file probability mass
        file_probabilities = torch.Tensor([score / sum(self.file_total_scores) for score in self.file_total_scores])

        number_of_samples_seen = sum(self.number_of_samples_per_file)
        target_num_samples = int(number_of_samples_seen * self.downsampled_batch_ratio / 100)

        if target_num_samples % self.batch_size == 1:
            # having a batch of just one element might be a problem for some operations (ex batch norm)
            target_num_samples -= 1

        self.grouped_samples_per_file = [0] * len(file_probabilities)

        self.normalizer = self.normalizer / number_of_samples_seen

        # oversample by a factor 1.25
        sampled_files = [
            el.item()
            for el in torch.multinomial(
                file_probabilities, num_samples=int(target_num_samples * 1.25), replacement=True
            )
        ]

        # ensure that we don't sample more points than available in a file
        counter_assigned = 0
        for file in sampled_files:
            if self.grouped_samples_per_file[file] < self.number_of_samples_per_file[file]:
                self.grouped_samples_per_file[file] += 1
                counter_assigned += 1
            if counter_assigned == target_num_samples:
                # we have selected all samples. The remaining are discarded
                break

    def get_samples_per_file(self, file_index: int) -> np.ndarray:
        """
        file-by-file sampling according to the probabilities computed above
        Args:
            file_index: index of the file

        Returns: a numpy ndarray of (index, weight)

        """
        assert file_index < len(self.number_of_samples_per_file)
        target_samples = self.grouped_samples_per_file[file_index]
        samples_list = np.empty(target_samples, dtype=np.dtype("i8,f8"))

        if target_samples == 0:
            return samples_list

        samples = self.scores_storage.get_trigger_samples(self.pipeline_id, self.trigger_id, file_index)
        sample_ids = [sample[0] for sample in samples]
        scores = torch.Tensor([sample[1] for sample in samples])

        selected_ids = torch.multinomial(scores, target_samples)

        for i in range(target_samples):
            samples_list[i] = (sample_ids[selected_ids[i]], self.normalizer / scores[selected_ids[i]])

        # this class is sequentially accessed so after the last file we no longer need the files
        if file_index == len(self.grouped_samples_per_file) - 1:
            self._clean_working_directory()

        return samples_list

    def _clean_working_directory(self) -> None:
        if os.path.isdir(TEMPORARY_LOCAL_STORAGE_PATH):
            shutil.rmtree(TEMPORARY_LOCAL_STORAGE_PATH)

    def _store_values(self) -> None:
        array_to_be_stored = np.array(self.to_be_stored, dtype=np.dtype("i8,f8"))
        self.scores_storage.save_trigger_sample(
            self.pipeline_id, self.trigger_id, self.current_file_index, array_to_be_stored, -1
        )
        self.file_total_scores.append(self.current_scores_sum)
        self.number_of_samples_per_file.append(len(self.to_be_stored))

        self.to_be_stored = []
        self.current_file_index += 1
        self.current_scores_sum = 0

    def get_total_number_of_files(self) -> int:
        return self.current_file_index
