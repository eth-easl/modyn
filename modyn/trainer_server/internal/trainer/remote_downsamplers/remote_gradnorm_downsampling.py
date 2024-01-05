import logging
from typing import Any, Optional

import torch
import logging
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)


class RemoteGradNormDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Method adapted from
    Not All Samples Are Created Equal: Deep Learning with Importance Sampling (Katharopoulos, Fleuret)
    The main idea is that the norm of the gradient up to the penultimate layer can be used to measure the importance
    of each sample. This is particularly convenient if the loss function is CrossEntropy since a closed form solution
    exists and thus no derivatives are needed (so it's marginally more expensive than computing the loss)
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        per_sample_loss: Any,
        device: str,
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, device)

        self.per_sample_loss_fct = per_sample_loss

        self.probabilities: list[torch.Tensor] = []
        self.number_of_points_seen = 0

    def get_scores(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss):
            # no need to autograd if cross entropy loss is used since closed form solution exists.
            with torch.inference_mode():
                # Because CrossEntropyLoss includes the softmax, we need to apply the
                # softmax to the forward output to obtain the probabilities
                probs = torch.nn.functional.softmax(forward_output, dim=1)
                num_classes = forward_output.shape[-1]

                # Pylint complains torch.nn.functional.one_hot is not callable for whatever reason
                one_hot_targets = torch.nn.functional.one_hot(  # pylint: disable=not-callable
                    target, num_classes=num_classes
                )
                scores = torch.norm(probs - one_hot_targets, dim=-1)
        else:
            sample_losses = self.per_sample_loss_fct(forward_output, target)
            last_layer_gradients = torch.autograd.grad(sample_losses.sum(), forward_output, retain_graph=False)[0]
            scores = torch.norm(last_layer_gradients, dim=-1)

        return scores.cpu()

    def init_downsampler(self) -> None:
        self.probabilities = []
        self.index_sampleid_map: list[int] = []
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        scores = self.get_scores(forward_output, target)
        self.probabilities.append(scores)
        self.number_of_points_seen += forward_output.shape[0]
        self.index_sampleid_map += sample_ids

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if len(self.probabilities) == 0:
            logger.warning("Empty probabilities, cannot select any points.")
            return [], torch.Tensor([])

        # select always at least 1 point
        target_size = max(int(self.downsampling_ratio * self.number_of_points_seen / 100), 1)

        probabilities = torch.cat(self.probabilities, dim=0)
        probabilities = probabilities / probabilities.sum()

        downsampled_idxs = torch.multinomial(probabilities, target_size, replacement=self.replacement)

        # lower probability, higher weight to reduce the variance
        weights = 1.0 / (self.number_of_points_seen * probabilities[downsampled_idxs])

        selected_ids = [self.index_sampleid_map[sample] for sample in downsampled_idxs]
        return selected_ids, weights
