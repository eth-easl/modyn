import logging
from collections import defaultdict
from typing import Any, List, Tuple

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)


class RemoteTokenUncertaintyDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Uncertainty-based down-sampling for generative models.
    Two modes:
      • token-level   (weight_per_sample=False) – pick individual tokens
      • sample-level  (weight_per_sample=True)  – pick whole samples
    """

    # ------------------------------------------------------------------ #
    #                         INITIALISATION                              #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        per_sample_loss: Any,
        device: str,
        generative: bool = True,
    ) -> None:
        super().__init__(
            pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device
        )

        if not generative:
            raise ValueError("generative must be True for token uncertainty sampling.")

        # ---------------- selector parameters ---------------- #
        self.score_metric: str = params_from_selector.get("score_metric", "Entropy")

        self.per_sample_top_k: int | None = params_from_selector.get("per_sample_top_k")
        self.per_sample_ratio: float | None = params_from_selector.get("per_sample_ratio")

        if self.per_sample_top_k is not None and self.per_sample_ratio is not None:
            raise ValueError("Specify *either* `per_sample_top_k` *or* `per_sample_ratio`, not both.")

        self.weight_per_sample: bool = params_from_selector.get("weight_per_sample", False)
        self.downsampling_ratio: float = params_from_selector.get("downsampling_ratio", 1.0)
        self.ratio_max: float = params_from_selector.get("ratio_max", 1.0)

        # ---------------- state buffers ---------------- #
        # one entry PER BATCH – concatenated on selection
        self._token_scores: list[torch.Tensor] = []      # neg-entropy values
        self._token_valid:  list[torch.Tensor] = []      # bool mask (True ⇒ target != -100)

        # one entry PER TOKEN – (sample_id, token_position)
        self.token_ids: list[Tuple[int, int]] = []
        self.device = device
    # ------------------------------------------------------------------ #
    #                           BOOK-KEEPING                              #
    # ------------------------------------------------------------------ #
    def init_downsampler(self) -> None:
        """Reset all internal buffers before a new training epoch/batch
        stream begins."""
        self._token_scores.clear()
        self._token_valid.clear()
        self.token_ids.clear()

    # ------------------------------------------------------------------ #
    #                          DATA INGESTION                             #
    # ------------------------------------------------------------------ #
    def inform_samples(
        self,
        sample_ids: List[int],
        forward_input: Any,
        forward_output: torch.Tensor,  # shape (B, T, V)
        target: torch.Tensor,          # shape (B, T)   (-100 marks padding)
        embedding: Any = None,
    ) -> None:
        """Store per-token negative entropies (–H) and validity flags."""
        B, T, _ = forward_output.shape

        # 1) calculate –entropy for every token
        with torch.no_grad():
            probs     = torch.softmax(forward_output, dim=2)          # p
            log_probs = (probs + 1e-6).log()                          # log p
            neg_entropy = (probs * log_probs).sum(dim=2)              # (B, T)

        # 2) flatten to 1-D
        flat_scores = neg_entropy.view(-1)                            # (B·T,)
        flat_valid  = (target.view(-1) != -100)                       # bool (B·T,)

        # 3) remember per-token ids (all tokens, valid or not)
        for i, sid in enumerate(sample_ids):
            for tok_pos in range(T):
                self.token_ids.append((sid, tok_pos))

        # 4) store tensors for later concatenation
        self._token_scores.append(flat_scores)
        self._token_valid.append(flat_valid)

    # ------------------------------------------------------------------ #
    #                       TOKEN / SAMPLE SELECTION                      #
    # ------------------------------------------------------------------ #
    def select_points(self) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """
        Returns
        -------
        token_ids : list[ (sample_id, token_pos) ]
            Same length and order as internal `token_ids`.
        weights   : torch.FloatTensor  (N,)
            1 → keep token, 0 → drop token.
            Invalid (target == -100) tokens are *always* 0.
        """
        num_tokens = len(self.token_ids)
        if num_tokens == 0:
            logger.warning("No tokens observed – returning empty selection")
            return [], torch.empty(0)

        # concat saved batches
        token_scores = torch.cat(self._token_scores, dim=0)           # (N,)
        token_valid  = torch.cat(self._token_valid,  dim=0)           # (N,)

        assert token_scores.numel() == num_tokens == token_valid.numel(), \
            "Internal buffer size mismatch."

        # mapping: sample_id -> list[token_idx]
        tok_idx_by_sample: dict[int, list[int]] = defaultdict(list)
        for idx, (sid, _) in enumerate(self.token_ids):
            tok_idx_by_sample[sid].append(idx)

        # mask we will fill with True for the items we keep
        keep_mask = torch.zeros(num_tokens, dtype=torch.bool)

        # ----------------------------------------------------------------
        # 1) SAMPLE-LEVEL mode – choose *samples*, keep all their tokens
        # ----------------------------------------------------------------
        if self.weight_per_sample:
            # score each sample by the mean neg-entropy of *valid* tokens
            sample_score = {
                sid: token_scores[idxs][token_valid[idxs]].mean()
                for sid, idxs in tok_idx_by_sample.items()
                if token_valid[idxs].any()        # ignore samples with 0 valid tokens
            }
            S = len(sample_score)

            if self.per_sample_top_k is not None:
                num_keep = min(self.per_sample_top_k, S)
            elif self.per_sample_ratio is not None:
                num_keep = max(int(self.per_sample_ratio * S / self.ratio_max), 1)
            else:
                num_keep = max(int(self.downsampling_ratio * S / self.ratio_max), 1)

            # choose samples with *lowest* neg-entropy (i.e. highest entropy)
            chosen_sids = sorted(sample_score, key=sample_score.get)[:num_keep]

            # mark all their tokens as kept – *except* invalid ones
            for sid in chosen_sids:
                for tok_idx in tok_idx_by_sample[sid]:
                    if token_valid[tok_idx]:
                        keep_mask[tok_idx] = True

        # ----------------------------------------------------------------
        # 2) TOKEN-LEVEL mode – pick individual tokens
        # ----------------------------------------------------------------
        else:
            if self.per_sample_top_k is not None or self.per_sample_ratio is not None:
                # --- per-sample quotas ----------------------------------
                for sid, idxs in tok_idx_by_sample.items():
                    valid_idxs = [i for i in idxs if token_valid[i]]
                    L = len(valid_idxs)
                    if L == 0:
                        continue

                    quota = (
                        min(self.per_sample_top_k, L)
                        if self.per_sample_top_k is not None
                        else max(int(self.per_sample_ratio * L / self.ratio_max), 1)
                    )

                    # take the `quota` most-uncertain valid tokens inside this sample
                    local_top = torch.argsort(token_scores[valid_idxs])[:quota]
                    keep_mask[torch.tensor(valid_idxs,device=self.device)[local_top]] = True
            else:
                # --- global quota ---------------------------------------
                valid_idxs = torch.arange(num_tokens, device=token_valid.device)[token_valid]
                quota = max(int(self.downsampling_ratio * valid_idxs.numel() / self.ratio_max), 1)

                global_top = torch.argsort(token_scores[valid_idxs])[:quota]
                keep_mask[valid_idxs[global_top]] = True

        # invalid tokens remain False → weight 0
        weights = keep_mask.float()
        selected_sample_ids: list[int] = []
        for idx, keep in enumerate(keep_mask):
            if keep:                         # token kept?
                sid, _ = self.token_ids[idx]
                if sid not in selected_sample_ids:
                    selected_sample_ids.append(sid)

        # Return those sample IDs together with the weight mask.
        return selected_sample_ids, weights
    # ------------------------------------------------------------------ #
    @property
    def requires_grad(self) -> bool:  # required by Modyn interface
        return False
