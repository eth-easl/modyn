from typing import Any

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class RemoteGradNormDownsampling(AbstractRemoteDownsamplingStrategy):
    def __init__(self, params_from_selector: dict, per_sample_loss: Any, num_classes: int = -1) -> None:
        super().__init__(params_from_selector)

        self.per_sample_loss_fct = per_sample_loss
        self.num_classes = num_classes

    def get_probabilities(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss) and self.num_classes > 0:
            # no need to autograd if cross entropy loss is used since closed form solution exists
            with torch.inference_mode():
                # Because CrossEntropyLoss includes the softmax, we need to apply the
                # softmax to the forward output to obtain the probabilities
                probs = torch.nn.functional.softmax(forward_output, dim=1)
                one_hot_targets = torch.nn.functional.one_hot(target, num_classes=self.num_classes)
                scores = torch.norm(probs - one_hot_targets, dim=-1)
        else:
            sample_losses = self.per_sample_loss_fct(forward_output, target)
            last_layer_gradients = torch.autograd.grad(sample_losses.sum(), forward_output, retain_graph=False)[0]
            scores = torch.norm(last_layer_gradients, dim=-1)

        return scores
