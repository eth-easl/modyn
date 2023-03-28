from typing import Union

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class RemoteLossDownsampling(AbstractRemoteDownsamplingStrategy):
    def __init__(self, model: torch.nn, params_from_selector: dict, params_from_trainer: dict) -> None:
        super().__init__(model, params_from_selector)

        assert "per_sample_loss_fct" in params_from_trainer
        self.per_sample_loss_fct = params_from_trainer["per_sample_loss_fct"]

    def get_probabilities_and_forward_outputs(
        self, data: Union[torch.Tensor, dict], target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(data)
        scores = self.per_sample_loss_fct(output, target).detach()
        probabilities = scores / scores.sum()
        return probabilities, output
