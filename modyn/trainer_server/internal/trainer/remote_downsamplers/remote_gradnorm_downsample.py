from typing import Any

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class RemoteGradNormDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Method adapted from
    Not All Samples Are Created Equal: Deep Learning with Importance Sampling (Katharopoulos, Fleuret)
    The main idea is that the norm of the gradient up to the penultimate layer can be used to measure the importance
    of each sample. This is particularly convenient if the loss function is CrossEntropy since a closed form solution
    exists and thus no derivatives are needed (so it's marginally more expensive than computing the loss)
    """

    def __init__(self, params_from_selector: dict, per_sample_loss: Any) -> None:
        super().__init__(params_from_selector)

        self.per_sample_loss_fct = per_sample_loss

    def get_probabilities(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss):
            # no need to autograd if cross entropy loss is used since closed form solution exists.
            with torch.inference_mode():
                # Because CrossEntropyLoss includes the softmax, we need to apply the
                # softmax to the forward output to obtain the probabilities
                probs = torch.nn.functional.softmax(forward_output, dim=1)
                num_classes = forward_output.shape[-1]
                one_hot_targets = torch.nn.functional.one_hot(target, num_classes=num_classes)
                scores = torch.norm(probs - one_hot_targets, dim=-1)
        else:
            sample_losses = self.per_sample_loss_fct(forward_output, target)
            last_layer_gradients = torch.autograd.grad(sample_losses.sum(), forward_output, retain_graph=False)[0]
            scores = torch.norm(last_layer_gradients, dim=-1)

        return scores
