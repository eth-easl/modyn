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
    def __init__(self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict) -> None:
        self.pipeline_id = pipeline_id
        self.batch_size = batch_size
        self.trigger_id = trigger_id

        assert "downsampled_batch_ratio" in params_from_selector
        self.downsampled_batch_ratio = params_from_selector["downsampled_batch_ratio"]
        self._sampling_concluded = False

        self.replacement = params_from_selector.get("replacement", True)

    def get_downsampled_batch_ratio(self) -> int:
        return self.downsampled_batch_ratio

    @abstractmethod
    def init_downsampler(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def inform_samples(self, forward_output: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
