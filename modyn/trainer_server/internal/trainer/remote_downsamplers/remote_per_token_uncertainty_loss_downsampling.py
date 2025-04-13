import logging
from typing import Any

import torch
import numpy as np

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)

class RemoteTokenUncertaintyDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Token-level uncertainty sampling for generative tasks.
    Each token in the target sequence has a predicted distribution p(vocab).
    We compute 'negative entropy' = sum_{v} [ p * log p ] at each token,
    then pick the tokens with the smallest negative entropy => largest real entropy.

    We expect 'generative=True' in the pipeline config. If it's not set,
    or set to False, we raise an error. Currently only 'Entropy' is implemented.

    Example config:
      selection_strategy:
        name: "RemoteTokenUncertaintyDownsampling"
        score_metric: "Entropy"
        generative: true
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        per_sample_loss: Any,  # not used
        device: str,
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        # For consistency with your pipeline config, we read 'generative'
        self.generative: bool = params_from_selector.get("generative", True)
        if not self.generative:
            raise ValueError(
                "RemoteTokenUncertaintyDownsampling only applies to generative tasks. "
                "Please set generative=True in your pipeline config."
            )

        self.score_metric = params_from_selector.get("score_metric", "Entropy")
        if self.score_metric != "Entropy":
            logger.warning("Only 'Entropy' is implemented in this token-level uncertainty strategy.")

        self.token_neg_entropies: list[torch.Tensor] = []
        self.token_ids: list[tuple[int, int]] = []  # (sample_id, token_idx)
        self.number_of_tokens_seen = 0

    def init_downsampler(self) -> None:
        self.token_neg_entropies = []
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
        forward_output shape: (B, T, V)
        target shape: (B, T), ignoring tokens with target == -100
        We compute token-level negative entropy for each valid token.
        """
        batch_size, seq_length, vocab_size = forward_output.shape

        with torch.no_grad():
            preds_3d = torch.nn.functional.softmax(forward_output, dim=2)
            log_preds_3d = (preds_3d + 1e-6).log()
            # negative entropy => shape (B, T)
            negent_2d = (log_preds_3d * preds_3d).sum(dim=2)

        # We'll store negative entropies for valid tokens
        # ignoring padding (target == -100).
        flat_negent = negent_2d.view(-1)  # shape (B*T,)
        index = 0
        for i in range(batch_size):
            for j in range(seq_length):
                if target[i, j] != -100:
                    self.token_ids.append((sample_ids[i], j))
                    self.number_of_tokens_seen += 1
                index += 1

        # store the entire flattened neg-entropy array to pick from later
        self.token_neg_entropies.append(flat_negent)

    def select_points(self) -> tuple[list[tuple[int, int]], torch.Tensor]:
        if self.number_of_tokens_seen == 0:
            logger.warning("No valid tokens. Returning empty.")
            return [], torch.Tensor([])

        # Concatenate all negative entropies => shape (# tokens total)
        all_negent = torch.cat(self.token_neg_entropies, dim=0)
        number_of_tokens = len(self.token_ids)

        if all_negent.shape[0] != number_of_tokens:
            logger.error("Mismatch: # token_ids != shape of neg-ent array.")
            return [], torch.Tensor([])

        target_size = max(int(self.downsampling_ratio * number_of_tokens / self.ratio_max), 1)
        # Sort ascending => smallest negative => largest real entropy
        sorted_indices = torch.argsort(all_negent)
        selected_indices = sorted_indices[:target_size]

        selected_tokens = [self.token_ids[idx] for idx in selected_indices.tolist()]
        # Uniform weights
        weights = torch.ones(target_size, dtype=torch.float)
        return selected_tokens, weights

    @property
    def requires_grad(self) -> bool:
        return False
