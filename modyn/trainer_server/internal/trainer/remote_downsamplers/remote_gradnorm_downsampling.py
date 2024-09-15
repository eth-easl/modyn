import logging
from typing import Any

import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
    unsqueeze_dimensions_if_necessary,
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
        modyn_config: dict,
        per_sample_loss: Any,
        device: str,
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        self.per_sample_loss_fct = per_sample_loss

        self.probabilities: list[torch.Tensor] = []
        self.number_of_points_seen = 0

    def init_downsampler(self) -> None:
        self.probabilities = []
        self.index_sampleid_map: list[int] = []
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        forward_output, target = unsqueeze_dimensions_if_necessary(forward_output, target)

        last_layer_gradients = self._compute_last_layer_gradient_wrt_loss_sum(
            self.per_sample_loss_fct, forward_output, target
        )
        # pylint: disable=not-callable
        scores = torch.linalg.vector_norm(last_layer_gradients, dim=1).cpu()
        self.probabilities.append(scores)
        self.number_of_points_seen += forward_output.shape[0]
        self.index_sampleid_map += sample_ids

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if len(self.probabilities) == 0:
            logger.warning("Empty probabilities, cannot select any points.")
            return [], torch.Tensor([])

        # select always at least 1 point
        target_size = max(int(self.downsampling_ratio * self.number_of_points_seen / self.ratio_max), 1)

        probabilities = torch.cat(self.probabilities, dim=0)
        probabilities = probabilities / probabilities.sum()

        downsampled_idxs = torch.multinomial(probabilities, target_size, replacement=False)

        # lower probability, higher weight to reduce the variance
        weights = 1.0 / (self.number_of_points_seen * probabilities[downsampled_idxs])

        selected_ids = [self.index_sampleid_map[sample] for sample in downsampled_idxs]
        return selected_ids, weights

    @property
    def requires_grad(self) -> bool:
        if isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss):
            return False

        return True
