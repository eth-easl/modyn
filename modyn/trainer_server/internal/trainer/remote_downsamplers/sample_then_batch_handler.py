import math
import shutil
from typing import Any

import numpy as np
import torch
from modyn.common.trigger_sample import TriggerSampleStorage
from typing_extensions import Self


class SampleThenBatchHandler:
    def __init__(self, pipeline_id: int, batch_size: int, downsampled_batch_ratio: float) -> None:
        # used to temporarily store the computed scores
        self.scores_storage = TriggerSampleStorage(".tmp_scores")
        self.current_pipeline_id = pipeline_id
        self.batch_size = batch_size
        self.downsampled_batch_ratio = downsampled_batch_ratio

        # below a list of arguments that are used throughout the function but not yet available
        self.file_total_scores: list[float] = []
        self.grouped_samples_per_file: list[int] = []
        self.number_of_samples_per_file: list[int] = []
        self.number_of_samples_seen = 0
        self._failed_accumulation = False

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
        self.number_of_samples_seen = 0
        return self

    def accumulate(self, sample_ids: list, scores: torch.Tensor, batch_number: int) -> None:
        """
        Function to accumulate ids and scores. These values are temporarily stored using TriggerSampleStorage
        Args:
            sample_ids: list of sample ids
            scores: tensor of scores
            batch_number: index of the batch
        """
        self.number_of_samples_seen += len(sample_ids)
        self.file_total_scores.append(scores.sum())
        self.number_of_samples_per_file.append(len(sample_ids))
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

        assert 0 < self.downsampled_batch_ratio < 1
        assert self.batch_size > 0

        # The aim is to an exact number of batches.
        # Virtually this number is self.downsampled_batch_ratio * self.batch_size
        # but this number is not ensured to be an integer
        num_samples = self.batch_size * math.floor(
            self.number_of_samples_seen * self.downsampled_batch_ratio / self.batch_size
        )
        self.grouped_samples_per_file = [0] * len(file_probabilities)

        counter_assigned = 0
        # oversample by a factor 1.25
        sampled_files = [
            el.item()
            for el in torch.multinomial(file_probabilities, num_samples=int(num_samples * 1.25), replacement=True)
        ]

        # ensure that we don't sample more points than available in a file
        for file in sampled_files:
            if self.grouped_samples_per_file[file] < self.number_of_samples_per_file[file]:
                self.grouped_samples_per_file[file] += 1
                counter_assigned += 1
            if counter_assigned == num_samples:
                # we have selected all samples. The remaining are discarded
                break

    def get_samples_per_file(self, file_index: int) -> np.ndarray:
        """
        file-by-file sampling according to the probabilities computed above
        Args:
            file_index: index of the file

        Returns: a numpy ndarray of (index, weight)

        """
        assert not self._failed_accumulation, "Something went wrong when probabilities were computed."
        target_samples = self.grouped_samples_per_file[file_index]
        samples_list = np.empty(target_samples, dtype=np.dtype("i8,f8"))

        if target_samples == 0:
            return samples_list

        samples = self.scores_storage.get_trigger_samples(self.current_pipeline_id, 0, file_index)
        print(target_samples, len(samples))
        sample_ids = [sample[0] for sample in samples]
        scores = torch.Tensor([sample[1] for sample in samples])
        selected_ids = torch.multinomial(scores, target_samples)

        for i in range(target_samples):
            samples_list[i] = (sample_ids[selected_ids[i]], scores[selected_ids[i]])

        if file_index == len(self.grouped_samples_per_file) - 1:
            shutil.rmtree(".tmp_scores")

        return samples_list
