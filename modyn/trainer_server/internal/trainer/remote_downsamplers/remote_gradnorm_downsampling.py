# remote_gradnorm_downsampling.py
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
    Adapted to handle generative tasks at sequence-level.
    If generative=True, forward_output is shape (B, T, V).
    We'll compute per-sample gradient norms by summing across tokens in the cross-entropy.
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
        generative: bool = False,
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        self.per_sample_loss_fct = per_sample_loss
        self.generative = generative

        self.probabilities: list[torch.Tensor] = []
        self.index_sampleid_map: list[int] = []
        self.number_of_points_seen = 0

    def init_downsampler(self) -> None:
        self.probabilities = []
        self.index_sampleid_map = []
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        # For classification or normal usage => shape (B, num_classes)
        # For generative => shape (B, T, V)
        # We'll assume per_sample_loss_fct can handle (B, T, V) => returns shape (B,) after summing across tokens.

        forward_output, target = unsqueeze_dimensions_if_necessary(forward_output, target)
        # => _compute_last_layer_gradient_wrt_loss_sum is the core function to get per-sample grads
        last_layer_gradients = self._compute_last_layer_gradient_wrt_loss_sum(
            self.per_sample_loss_fct, forward_output, target
        )
        # scores = L2 norm of the gradient => shape (B,)
        scores = torch.linalg.vector_norm(last_layer_gradients, dim=1).cpu()
        self.probabilities.append(scores)
        self.number_of_points_seen += forward_output.shape[0]
        self.index_sampleid_map += sample_ids

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if len(self.probabilities) == 0:
            logger.warning("Empty probabilities, cannot select any points.")
            return [], torch.Tensor([])

        target_size = max(int(self.downsampling_ratio * self.number_of_points_seen / self.ratio_max), 1)

        probabilities = torch.cat(self.probabilities, dim=0)  # shape (#samples, )
        probabilities = probabilities / probabilities.sum()

        downsampled_idxs = torch.multinomial(probabilities, target_size, replacement=False)
        weights = 1.0 / (self.number_of_points_seen * probabilities[downsampled_idxs])

        selected_ids = [self.index_sampleid_map[sample] for sample in downsampled_idxs]
        return selected_ids, weights

    @property
    def requires_grad(self) -> bool:
        # If your loss is cross entropy, no need for grad
        if isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss):
            return False
        return True
