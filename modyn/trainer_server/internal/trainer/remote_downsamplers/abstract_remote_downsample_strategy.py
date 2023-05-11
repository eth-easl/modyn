from abc import ABC, abstractmethod
from typing import Union

import torch


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
    def __init__(self, params_from_selector: dict) -> None:
        assert "sample_before_batch" in params_from_selector
        self.sample_before_batch = params_from_selector["sample_before_batch"]

        if self.sample_before_batch:
            assert "downsampled_batch_ratio" in params_from_selector
            self.downsampled_batch_ratio = params_from_selector["downsampled_batch_ratio"]
        else:
            assert "downsampled_batch_size" in params_from_selector
            self.downsampled_batch_size = params_from_selector["downsampled_batch_size"]

        self.replacement = params_from_selector.get("replacement", True)

    def sample(
        self,
        forward_output: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = self.get_probabilities(forward_output, target)
        downsampled_idxs = torch.multinomial(probabilities, self.downsampled_batch_size, replacement=self.replacement)

        # lower probability, higher weight to reducce the variance
        weights = 1.0 / (forward_output.shape[0] * probabilities[downsampled_idxs])

        return downsampled_idxs, weights

    @abstractmethod
    def get_probabilities(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
