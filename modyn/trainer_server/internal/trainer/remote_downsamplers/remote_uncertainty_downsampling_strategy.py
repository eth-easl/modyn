import logging
from typing import Any

import numpy as np
import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_per_label_remote_downsample_strategy import (
    AbstractPerLabelRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    unsqueeze_dimensions_if_necessary,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.shuffling import _shuffle_list_and_tensor

logger = logging.getLogger(__name__)


class RemoteUncertaintyDownsamplingStrategy(AbstractPerLabelRemoteDownsamplingStrategy):
    """
    Strategy introduced in:
      *Selection via Proxy: Efficient Data Selection for Deep Learning*
    Implementation adapted from:
      *DEEPCORE* https://github.com/PatrickZH/DeepCore (uncertainty.py)

    This strategy collects a measure of uncertainty (LeastConfidence, Entropy or Margin) for each *sample*
    and selects the top-k most uncertain samples.

    For *generative tasks* (sequence -> sequence), set `generative=True` in your pipeline config, so if the
    model output is (B, T, V), we compute a single score per sample by aggregating (negative) entropy across T.

    The user can specify the metric via `score_metric`: one of ["LeastConfidence", "Entropy", "Margin"].
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
        generative: bool = True,
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        self.criterion = per_sample_loss  # not really used here
        self.scores = np.array([])

        if "score_metric" not in params_from_selector:
            raise ValueError("Please provide a 'score_metric': LeastConfidence, Entropy, or Margin.")
        self.score_metric = params_from_selector["score_metric"]

        self.generative = generative
        # Balanced selection across classes?
        self.balance = params_from_selector["balance"]
        if self.balance:
            # If the pipeline calls us label by label, we store partial results and unify
            self.already_selected_ids: list[int] = []
            self.already_selected_weights = torch.tensor([]).float()
            self.requires_data_label_by_label = True
        else:
            self.requires_data_label_by_label = False

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        assert embedding is None
        if hasattr(forward_output, "logits"):
            forward_output = forward_output.logits
        forward_output, _ = unsqueeze_dimensions_if_necessary(forward_output, target)
        these_scores = self._compute_score(forward_output.detach())
        self.scores = np.append(self.scores, these_scores)
        self.index_sampleid_map += sample_ids

    def _compute_score(self, forward_output: torch.Tensor, disable_softmax: bool = False) -> np.ndarray:
        """
        * Classification case: shape (B, num_classes)
        * Generative case:    shape (B, T, vocab)
            - We'll do only 'Entropy' here by summing token-level negative entropies => single sample-level score.
        """

        if self.generative:
            # For demonstration, we only implement 'Entropy' in the generative case

            if self.score_metric != "Entropy":
                raise ValueError(f"In generative mode, only 'Entropy' is supported. Got {self.score_metric}.")

            # forward_output shape: (B, T, V)
            batch_size, seq_length, vocab_size = forward_output.shape
            if not disable_softmax:
                preds_3d = torch.nn.functional.softmax(forward_output, dim=2)
            else:
                preds_3d = forward_output

            log_preds_3d = (preds_3d + 1e-6).log()
            # negative entropy = sum( p log p ), so shape => (B, T)
            per_token_negent = (log_preds_3d * preds_3d).sum(dim=2)
            # sum across T => shape (B,) => single negative entropy score for each sample
            scores = per_token_negent.sum(dim=1)
            return scores.cpu().numpy()

        # Non-generative: standard classification approach
        feature_size = forward_output.size(1)

        if self.score_metric == "LeastConfidence":
            if feature_size == 1:
                # binary classification
                scores = torch.abs(forward_output).squeeze(1).cpu().numpy()
            else:
                scores = forward_output.max(dim=1).values.cpu().numpy()

        else:
            # if feature_size == 1 => binary classification => treat as sigmoid
            if feature_size == 1:
                preds = torch.sigmoid(forward_output) if not disable_softmax else forward_output
                preds = torch.cat((1 - preds, preds), dim=1)
            else:
                preds = torch.nn.functional.softmax(forward_output, dim=1) if not disable_softmax else forward_output

            if self.score_metric == "Entropy":
                scores = (np.log(preds + 1e-6) * preds).sum(axis=1)
            elif self.score_metric == "Margin":
                preds_argmax = torch.argmax(preds, dim=1)
                max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                preds_sub_argmax = torch.argmax(preds, dim=1)
                second_max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]
                scores = (max_preds - second_max_preds).cpu().numpy()
            else:
                raise AssertionError(f"Unknown metric {self.score_metric} in classification mode.")

        return scores

    def inform_end_of_current_label(self) -> None:
        # For balanced selection => pick from each label separately
        assert self.balance
        selected_samples, selected_weights = self._select_from_scores()
        self.already_selected_ids += selected_samples
        self.already_selected_weights = torch.cat((self.already_selected_weights, selected_weights))

        self.scores = np.array([])
        self.index_sampleid_map: list[int] = []

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if self.balance:
            ids, weights = self.already_selected_ids, self.already_selected_weights
        else:
            ids, weights = self._select_from_scores()
        return _shuffle_list_and_tensor(ids, weights)

    def _select_from_scores(self) -> tuple[list[int], torch.Tensor]:
        number_of_samples = len(self.scores)
        target_size = max(int(self.downsampling_ratio * number_of_samples / self.ratio_max), 1)
        selected_indices, weights = self._select_indexes_from_scores(target_size)
        selected_ids = [self.index_sampleid_map[index] for index in selected_indices]
        return selected_ids, weights

    def init_downsampler(self) -> None:
        self.scores = []
        self.index_sampleid_map = []

    def _select_indexes_from_scores(self, target_size: int) -> tuple[list[int], torch.Tensor]:
        # ascending order => picking smallest negative => highest actual uncertainty
        indices = np.argsort(self.scores)[:target_size].tolist()
        weights = torch.ones(target_size).float()
        return indices, weights

    @property
    def requires_grad(self) -> bool:
        return False
