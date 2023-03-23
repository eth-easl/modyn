from typing import Tuple, Union

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class RemoteLossDownsampling(AbstractRemoteDownsamplingStrategy):
    def __init__(self, params_from_selector: dict, params_from_trainer: dict) -> None:
        super().__init__(params_from_selector, params_from_trainer)

        assert "per_sample_loss_fct" in params_from_trainer
        self.per_sample_loss_fct = params_from_trainer["per_sample_loss_fct"]

    def sample(
        self, data: Union[torch.Tensor, dict], target: torch.Tensor, sample_ids: list
    ) -> Tuple[Union[torch.Tensor, dict], torch.Tensor, torch.Tensor, list]:
        output = self.model(data)
        scores = self.per_sample_loss_fct(output, target).detach()

        # sample according the score distribution
        probabilities = scores / scores.sum()

        downsampled_idxs = torch.multinomial(probabilities, self.downsampled_batch_size, replacement=True)

        weights = 1.0 / (len(target) * probabilities[downsampled_idxs])

        selected_sample_ids = [sample_ids[i] for i in downsampled_idxs]

        return data[downsampled_idxs], weights, target[downsampled_idxs], selected_sample_ids
