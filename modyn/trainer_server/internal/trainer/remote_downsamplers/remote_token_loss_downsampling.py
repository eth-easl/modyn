import logging
from typing import Any

import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)


class RemoteTokenLossDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Per-token loss-based downsampling strategy. Tokens with higher loss have higher chance to be selected.
    Inspired by per-token importance sampling.
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        per_sample_loss: Any,  # unused
        device: str,
        generative: bool = True,  # always per-token for this class
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)
        self.token_losses: list[torch.Tensor] = []
        self.token_ids: list[tuple[int, int]] = []  # (sample_id, token_position)
        self.token_mask: list[bool] = []
        self.number_of_tokens_seen = 0

    def get_token_scores(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if hasattr(forward_output, "logits"):
            logits = forward_output.logits
        else:
            logits = forward_output

        batch_size, seq_length, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target.view(-1)

        token_losses = torch.nn.functional.cross_entropy(
            logits_flat, target_flat, reduction="none", ignore_index=-100
        )  # shape: (batch_size * seq_length)

        return token_losses.detach().view(batch_size, seq_length), target

    def init_downsampler(self) -> None:
        self.token_losses = []
        self.token_ids = []
        self.token_mask = []
        self.number_of_tokens_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        token_losses, full_target = self.get_token_scores(forward_output, target)
        batch_size, seq_length = token_losses.shape

        self.token_losses.append(token_losses.flatten())
        for i in range(batch_size):
            for j in range(seq_length):
                if full_target[i, j] != -100:
                    self.token_ids.append((sample_ids[i], j))
                    self.token_mask.append(True)
                    self.number_of_tokens_seen += 1

    def select_points(self) -> tuple[list[tuple[int, int]], torch.Tensor]:
        if self.number_of_tokens_seen == 0:
            logger.warning("No valid tokens seen, cannot select any.")
            return [], torch.Tensor([])

        losses = torch.cat(self.token_losses)[torch.tensor(self.token_mask)]
        probs = losses / losses.sum()

        target_size = max(int(self.downsampling_ratio * self.number_of_tokens_seen / self.ratio_max), 1)
        selected_indices = torch.multinomial(probs, target_size, replacement=False)
        selected_tokens = [self.token_ids[i] for i in selected_indices.tolist()]

        weights = 1.0 / (self.number_of_tokens_seen * probs[selected_indices])
        return selected_tokens, weights

    @property
    def requires_grad(self) -> bool:
        return False
