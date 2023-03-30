from typing import Any

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class RemoteLossDownsampling(AbstractRemoteDownsamplingStrategy):
    def __init__(self, params_from_selector: dict, per_sample_loss: Any) -> None:
        super().__init__(params_from_selector)

        self.per_sample_loss_fct = per_sample_loss

    def get_probabilities(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = self.per_sample_loss_fct(forward_output, target).detach()
        probabilities = scores / scores.sum()
        return probabilities
