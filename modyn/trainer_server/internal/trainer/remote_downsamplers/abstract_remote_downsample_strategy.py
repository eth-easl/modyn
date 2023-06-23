from abc import ABC, abstractmethod
from typing import Union

import torch


def get_tensors_subset(
    selected_indexes: list[int], data: Union[torch.Tensor, dict], target: torch.Tensor, sample_ids: list
) -> tuple[Union[torch.Tensor, dict], torch.Tensor]:
    # Starting from the batch, we want to keep only the selected samples

    # first of all we compute the position of each selected index within the batch
    in_batch_index = [sample_ids.index(selected_index) for selected_index in selected_indexes]

    # then we extract the data
    if isinstance(data, torch.Tensor):
        sub_data = data[in_batch_index]
    elif isinstance(data, dict):
        sub_data = {key: tensor[in_batch_index] for key, tensor in data.items()}

    # and the targets
    sub_target = target[in_batch_index]

    return sub_data, sub_target


class AbstractRemoteDownsamplingStrategy(ABC):
    def __init__(self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict) -> None:
        self.pipeline_id = pipeline_id
        self.batch_size = batch_size
        self.trigger_id = trigger_id

        assert "downsampling_ratio" in params_from_selector
        self.downsampling_ratio = params_from_selector["downsampling_ratio"]
        self._sampling_concluded = False

        self.replacement = params_from_selector.get("replacement", True)
        self.index_sampleid_map: list[int] = []

    def get_downsampling_ratio(self) -> int:
        return self.downsampling_ratio

    @abstractmethod
    def init_downsampler(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def inform_samples(self, sample_ids: list[int], forward_output: torch.Tensor, target: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def select_points(self) -> tuple[list[int], torch.Tensor]:
        raise NotImplementedError
