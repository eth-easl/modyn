import logging
from typing import Any, List, Tuple
from collections import defaultdict

import torch
import numpy as np

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)

class RemoteTokenUncertaintyDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Token- or sample-level uncertainty sampling for generative tasks.

    Configurable params (in params_from_selector):
      - generative: bool (must be True)
      - score_metric: "Entropy"
      - per_sample_top_k: int (optional)
      - per_sample_ratio: float (optional, fraction of tokens per sample or of samples if weight_per_sample=True)
      - downsampling_ratio: float (fallback global fraction)
      - ratio_max: float (fallback denominator for global ratio)
      - weight_per_sample: bool (if True, select at sample-level; else per-token)
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

        # Must be generative
        self.generative: bool = params_from_selector.get("generative", True)
        if not self.generative:
            raise ValueError(
                "RemoteTokenUncertaintyDownsampling only applies to generative tasks. "
                "Please set generative=True in your pipeline config."
            )

        # Only entropy supported
        self.score_metric = params_from_selector.get("score_metric", "Entropy")
        if self.score_metric != "Entropy":
            logger.warning("Only 'Entropy' is implemented in this token-level uncertainty strategy.")

        # Per-sample selection parameters (mutually exclusive)
        self.per_sample_top_k = params_from_selector.get("per_sample_top_k", None)
        self.per_sample_ratio = params_from_selector.get("per_sample_ratio", None)
        if self.per_sample_top_k is not None and self.per_sample_ratio is not None:
            raise ValueError("Specify only one of 'per_sample_top_k' or 'per_sample_ratio'.")

        # Toggle between token- and sample-level weighting
        self.weight_per_sample = params_from_selector.get("weight_per_sample", False)

        # Data storage
        self.token_neg_entropies: List[torch.Tensor] = []
        self.token_ids: List[Tuple[int, int]] = []  # (sample_id, token_idx)
        self.number_of_tokens_seen = 0

    def init_downsampler(self) -> None:
        self.token_neg_entropies = []
        self.token_ids = []
        self.number_of_tokens_seen = 0

    def inform_samples(
        self,
        sample_ids: List[int],
        forward_input: Any,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Any = None,
    ) -> None:
        batch_size, seq_length, vocab_size = forward_output.shape
        with torch.no_grad():
            preds = torch.nn.functional.softmax(forward_output, dim=2)
            log_preds = (preds + 1e-6).log()
            negent = (log_preds * preds).sum(dim=2)  # shape (B, T)

        flat_negent = negent.view(-1)  # shape (B*T,)
        for i in range(batch_size):
            for j in range(seq_length):
                if target[i, j] != -100:
                    self.token_ids.append((sample_ids[i], j))
                    self.number_of_tokens_seen += 1

        self.token_neg_entropies.append(flat_negent)

    def select_points(self) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        N = len(self.token_ids)
        if N == 0:
            logger.warning("No valid tokens. Returning empty.")
            return [], torch.Tensor([])

        all_negent = torch.cat(self.token_neg_entropies, dim=0)
        if all_negent.shape[0] != N:
            logger.error("Mismatch: # token_ids != shape of neg-ent array.")
            return [], torch.Tensor([])

        # Group token indices by sample
        sample_to_idxs = defaultdict(list)
        for idx, (sample_id, _) in enumerate(self.token_ids):
            sample_to_idxs[sample_id].append(idx)

        selected_mask = torch.zeros(N, dtype=torch.bool)

        if self.weight_per_sample:
            # SAMPLE-LEVEL: average each sample’s negent, pick k samples, then mark all tokens
            sample_scores = {
                sid: all_negent[idxs].mean()
                for sid, idxs in sample_to_idxs.items()
            }
            S = len(sample_scores)
            if self.per_sample_top_k is not None:
                K = min(self.per_sample_top_k, S)
            elif self.per_sample_ratio is not None:
                K = max(int(self.per_sample_ratio * S / self.ratio_max), 1)
            else:
                K = max(int(self.downsampling_ratio * S / self.ratio_max), 1)
            # lowest mean = most uncertain
            top_sids = sorted(sample_scores, key=sample_scores.get)[:K]
            selected_sids = set(top_sids)
            for idx, (sid, _) in enumerate(self.token_ids):
                if sid in selected_sids:
                    selected_mask[idx] = True

        else:
            # TOKEN-LEVEL: as before
            if self.per_sample_top_k is not None or self.per_sample_ratio is not None:
                for idxs in sample_to_idxs.values():
                    L = len(idxs)
                    k = (
                        min(self.per_sample_top_k, L)
                        if self.per_sample_top_k is not None
                        else max(int(self.per_sample_ratio * L / self.ratio_max), 1)
                    )
                    group_scores = all_negent[idxs]
                    topk = torch.argsort(group_scores)[:k]
                    for s in topk.tolist():
                        selected_mask[idxs[s]] = True
            else:
                target_size = max(int(self.downsampling_ratio * N / self.ratio_max), 1)
                topk_global = torch.argsort(all_negent)[:target_size]
                selected_mask[topk_global] = True

        weights = selected_mask.float()
        return self.token_ids.copy(), weights

    @property
    def requires_grad(self) -> bool:
        return False
