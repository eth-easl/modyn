# remote_token_gradnorm_downsampling.py
import logging
from typing import Any

import numpy as np
import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)


class RemoteTokenGradNormDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Token-level GradNorm approach for generative tasks.
    We do a partial backward pass for each token individually to get the gradient norm. (Very expensive!)
    Each token is treated as a separate 'sample'.
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
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)
        self.per_sample_loss_fct = per_sample_loss
        self.generative: bool = params_from_selector.get("generative", True)
        self.token_gradnorms: list[float] = []
        self.token_ids: list[tuple[int, int]] = []
        self.number_of_tokens_seen = 0

    def init_downsampler(self) -> None:
        self.token_gradnorms = []
        self.token_ids = []
        self.number_of_tokens_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        """
        forward_output: (B, T, V)
        We do a partial backward pass for each token. This is extremely slow.
        """
        batch_size, seq_length, vocab_size = forward_output.shape

        for i in range(batch_size):
            # We'll do separate forward/backward for each token:
            # Not recommended for large T => O(B*T) backward passes.
            for j in range(seq_length):
                if target[i, j] == -100:
                    continue  # skip padded tokens
                # Single-token loss => forward_output[i, j, :] => shape (V,)
                # We'll build a tiny shape(1,V) input for that token
                token_logits = forward_output[i, j, :].unsqueeze(0)  # shape(1,V)
                token_target = target[i, j].unsqueeze(0)  # shape(1,)

                # Clear grads
                token_logits.requires_grad_(True)
                loss = self.per_sample_loss_fct(token_logits, token_target)
                # partial backward
                grads = torch.autograd.grad(loss, token_logits, retain_graph=True)[0]  # shape(1,V)
                gradnorm = torch.linalg.vector_norm(grads, dim=1).item()
                self.token_gradnorms.append(gradnorm)
                self.token_ids.append((sample_ids[i], j))
                self.number_of_tokens_seen += 1

    def select_points(self) -> tuple[list[tuple[int, int]], torch.Tensor]:
        if self.number_of_tokens_seen == 0:
            logger.warning("No valid tokens. Returning empty.")
            return [], torch.Tensor([])

        token_gradnorms_np = np.array(self.token_gradnorms)
        # Normalize => sum(token_gradnorms) to get probabilities
        denom = token_gradnorms_np.sum()
        if denom < 1e-12:
            logger.warning("Sum of GradNorm is near zero. Returning random selection.")
            token_gradnorms_np = np.ones_like(token_gradnorms_np) / len(token_gradnorms_np)
        else:
            token_gradnorms_np = token_gradnorms_np / denom

        target_size = max(int(self.downsampling_ratio * self.number_of_tokens_seen / self.ratio_max), 1)
        selected_indices = np.random.choice(
            len(token_gradnorms_np), size=target_size, replace=False, p=token_gradnorms_np
        )

        weights = 1.0 / (self.number_of_tokens_seen * token_gradnorms_np[selected_indices])
        selected_tokens = [self.token_ids[idx] for idx in selected_indices]
        return selected_tokens, torch.tensor(weights, dtype=torch.float)

    @property
    def requires_grad(self) -> bool:
        return True
