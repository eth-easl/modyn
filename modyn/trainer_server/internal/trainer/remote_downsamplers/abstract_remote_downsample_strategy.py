from typing import Tuple, Union

import torch


class AbstractRemoteDownsamplingStrategy:
    def __init__(self, params_from_selector: dict, params_from_trainer: dict) -> None:
        assert "model" in params_from_trainer
        self.model = params_from_trainer["model"]

        assert "downsampled_batch_size" in params_from_selector
        self.downsampled_batch_size = params_from_selector["downsampled_batch_size"]

    def sample(
        self, data: Union[torch.Tensor, dict], target: torch.Tensor, sample_ids: list
    ) -> Tuple[Union[torch.Tensor, dict], torch.Tensor, torch.Tensor, list, torch.Tensor]:
        raise NotImplementedError()
