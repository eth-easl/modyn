import math
import os.path
import shutil
from typing import Any

import numpy as np
import torch
from modyn.common.trigger_sample import TriggerSampleStorage
from typing_extensions import Self

TEMPORARY_LOCAL_STORAGE_PATH = ".tmp_scores"  # should we provide it through the config?


class SampleThenBatchHandler:
    def __init__(self, pipeline_id: int, batch_size: int, downsampled_batch_ratio: float) -> None:
        # used to temporarily store the computed scores
        self.scores_storage = TriggerSampleStorage(TEMPORARY_LOCAL_STORAGE_PATH)
        self.current_pipeline_id = pipeline_id
        assert batch_size > 0
        self.batch_size = batch_size
        assert 0 < downsampled_batch_ratio < 1
        self.downsampled_batch_ratio = downsampled_batch_ratio

        # below a list of arguments that are used throughout the function but not yet available
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
        assert self.current_pipeline_id >= 0
        assert 0 < self.downsampled_batch_ratio < 1
        self.file_total_scores = []
        self.number_of_samples_per_file = []
        self.normalizer = 0.0
        self._clean_working_directory()
        return self

    def accumulate(self, sample_ids: list, scores: torch.Tensor, batch_number: int) -> None:
        """
        Function to accumulate ids and scores. These values are temporarily stored using TriggerSampleStorage
        Args:
            sample_ids: list of sample ids
            scores: tensor of scores
            batch_number: index of the batch
        """
        assert len(sample_ids) == scores.shape[0]

        self.file_total_scores.append(scores.sum().item())
        self.number_of_samples_per_file.append(len(sample_ids))
        self.normalizer += scores.sum().item()
        to_save = np.empty(len(sample_ids), dtype=np.dtype("i8,f8"))
        for i, (sample_id, score) in enumerate(zip(sample_ids, scores)):
            to_save[i] = (sample_id, score)
        self.scores_storage.save_trigger_sample(self.current_pipeline_id, 0, batch_number, to_save, -1)

    def __exit__(self, *exc: Any) -> None:
        """
        Used in the context manager.
        When all datapoints have been accumulated, the number of points per file are sampled.
        """
        assert exc[0] is None, f"Something went wrong: {exc}"

        file_probabilities = torch.Tensor([score / sum(self.file_total_scores) for score in self.file_total_scores])

        # The aim is to an exact number of batches.
        # Virtually this number is self.downsampled_batch_ratio * self.batch_size
        # but this number is not ensured to be an integer
        number_of_samples_seen = sum(self.number_of_samples_per_file)
        target_num_samples = self.batch_size * math.floor(
            number_of_samples_seen * self.downsampled_batch_ratio / self.batch_size
        )

        self.grouped_samples_per_file = [0] * len(file_probabilities)

        self.normalizer = self.normalizer / number_of_samples_seen

        counter_assigned = 0
        # oversample by a factor 1.25
        sampled_files = [
            el.item()
            for el in torch.multinomial(
                file_probabilities, num_samples=int(target_num_samples * 1.25), replacement=True
            )
        ]

        # ensure that we don't sample more points than available in a file
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

        samples = self.scores_storage.get_trigger_samples(self.current_pipeline_id, 0, file_index)
        sample_ids = [sample[0] for sample in samples]
        scores = torch.Tensor([sample[1] for sample in samples])

        selected_ids = torch.multinomial(scores, target_samples)

        for i in range(target_samples):
            samples_list[i] = (sample_ids[selected_ids[i]], self.normalizer / scores[selected_ids[i]])

        if file_index == len(self.grouped_samples_per_file) - 1:
            self._clean_working_directory()

        return samples_list

    def _clean_working_directory(self) -> None:
        if os.path.isdir(TEMPORARY_LOCAL_STORAGE_PATH):
            shutil.rmtree(TEMPORARY_LOCAL_STORAGE_PATH)
