import logging
from typing import Any

import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)


class RemoteLossDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Method inspired by
    Not All Samples Are Created Equal: Deep Learning with Importance Sampling (Katharopoulos, Fleuret)
    Instead of computing the last layer gradient (as GradNorm does), here, the selection proxy is the loss. Hence,
    a higher loss means a higher probability of being selected. This version is cheaper but less accurate.
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
        self.probabilities: list[torch.Tensor] = []
        self.number_of_points_seen = 0
        self.generative = generative

    def get_scores(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Old behavior: for non-generative tasks use the provided per_sample_loss function
        if not self.generative:
            scores = self.per_sample_loss_fct(forward_output, target).detach()
            return scores

        # New behavior: for generative tasks, compute per-sample loss by averaging token losses while ignoring -100 tokens
        if hasattr(forward_output, "logits"):
            logits = forward_output.logits
        else:
            logits = forward_output

        batch_size, seq_length, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        target_flat = target.reshape(-1)
        token_losses = torch.nn.functional.cross_entropy(logits_flat, target_flat, reduction="none", ignore_index=-100)
        token_losses = token_losses.view(batch_size, seq_length)
        mask = (target != -100).float()
        per_sample_loss = torch.sum(token_losses, dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return per_sample_loss.detach()

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
        scores = self.get_scores(forward_output, target)

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
        return False
