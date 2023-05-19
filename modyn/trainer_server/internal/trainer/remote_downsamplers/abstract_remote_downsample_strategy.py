from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.sample_then_batch_handler import SampleThenBatchHandler


def get_tensors_subset(
    indexes: torch.Tensor, data: Union[torch.Tensor, dict], target: torch.Tensor, sample_ids: list
) -> tuple[Union[torch.Tensor, dict], torch.Tensor, list]:
    if isinstance(data, torch.Tensor):
        sub_data = data[indexes]
    elif isinstance(data, dict):
        sub_data = {key: tensor[indexes] for key, tensor in data.items()}

    sub_target = target[indexes]
    sub_sample_ids = [sample_ids[i] for i in indexes]

    return sub_data, sub_target, sub_sample_ids


class AbstractRemoteDownsamplingStrategy(ABC):
    def __init__(self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict) -> None:
        assert "sample_before_batch" in params_from_selector
        self.sample_before_batch = params_from_selector["sample_before_batch"]
        self.pipeline_id = pipeline_id
        self.batch_size = batch_size
        self.trigger_id = trigger_id

        if self.sample_before_batch:
            assert "downsampled_batch_ratio" in params_from_selector
            self.downsampled_batch_ratio = params_from_selector["downsampled_batch_ratio"]
            assert "maximum_keys_in_memory" in params_from_selector
            self.sample_then_batch_handler = SampleThenBatchHandler(
                self.pipeline_id,
                self.trigger_id,
                self.batch_size,
                self.downsampled_batch_ratio,
                params_from_selector["maximum_keys_in_memory"],
            )
        else:
            assert "downsampled_batch_size" in params_from_selector
            self.downsampled_batch_size = params_from_selector["downsampled_batch_size"]

        self.replacement = params_from_selector.get("replacement", True)

    def get_downsampled_batch_ratio(self) -> int:
        assert self.sample_before_batch
        return self.downsampled_batch_ratio

    def batch_then_sample(
        self,
        forward_output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert not self.sample_before_batch
        scores = self.get_scores(forward_output, target)
        probabilities = scores / scores.sum()
        downsampled_idxs = torch.multinomial(probabilities, self.downsampled_batch_size, replacement=self.replacement)

        # lower probability, higher weight to reducce the variance
        weights = 1.0 / (forward_output.shape[0] * probabilities[downsampled_idxs])

        return downsampled_idxs, weights

    def get_sample_then_batch_accumulator(self) -> SampleThenBatchHandler:
        assert self.sample_before_batch
        return self.sample_then_batch_handler

    def get_samples_for_file(self, file_index: int) -> np.ndarray:
        assert self.sample_before_batch
        return self.sample_then_batch_handler.get_samples_per_file(file_index)

    @abstractmethod
    def get_scores(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
