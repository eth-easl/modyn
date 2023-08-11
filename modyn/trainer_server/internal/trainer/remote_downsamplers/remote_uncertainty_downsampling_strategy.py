from typing import Any, Optional

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_per_label_remote_downsample_strategy import (
    AbstractPerLabelRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.shuffling import _shuffle_list_and_tensor


class RemoteUncertaintyDownsamplingStrategy(AbstractPerLabelRemoteDownsamplingStrategy):
    def __init__(
        self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict, per_sample_loss: Any
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector)

        self.criterion = per_sample_loss
        self.scores = np.array([])

        if "score_metric" not in params_from_selector:
            raise ValueError(
                "Please provide a way to score uncertainty. Options available: LeastConfidence, Entropy, Margin"
            )
        self.score_metric = params_from_selector["score_metric"]

        # if true, the downsampling is balanced across classes ex class sizes = [10, 50, 30] and 50% downsampling
        # yields the following downsampled class sizes [5, 25, 15] while without balance something like [0, 45, 0] can
        # happen
        self.balance = params_from_selector["balance"]
        if self.balance:
            self.already_selected_samples: list[int] = []
            self.already_selected_weights = torch.tensor([]).float()
            self.requires_data_label_by_label = True
        else:
            self.requires_data_label_by_label = False

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        assert embedding is None

        self.scores = np.append(self.scores, self._compute_score(forward_output.detach()))
        # keep the mapping index<->sample_id
        self.index_sampleid_map += sample_ids

    def _compute_score(self, forward_output: torch.Tensor) -> np.ndarray:
        if self.score_metric == "LeastConfidence":
            scores = forward_output.max(dim=1).values.cpu().numpy()
        elif self.score_metric == "Entropy":
            preds = torch.nn.functional.softmax(forward_output, dim=1).cpu().numpy()
            scores = (np.log(preds + 1e-6) * preds).sum(axis=1)
        elif self.score_metric == "Margin":
            preds = torch.nn.functional.softmax(forward_output, dim=1)
            preds_argmax = torch.argmax(preds, dim=1)
            max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
            preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
            preds_sub_argmax = torch.argmax(preds, dim=1)
            scores = (max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy()
        else:
            raise AssertionError("The required metric does not exist")

        return scores

    def inform_end_of_current_label(self) -> None:
        assert self.balance
        selected_samples, selected_weights = self._select_from_scores()
        self.already_selected_samples += selected_samples
        self.already_selected_weights = torch.cat((self.already_selected_weights, selected_weights))
        self.scores = np.array([])
        self.index_sampleid_map: list[int] = []

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if self.balance:
            return _shuffle_list_and_tensor(self.already_selected_samples, self.already_selected_weights)
        return self._select_from_scores()

    def _select_from_scores(self) -> tuple[list[int], torch.Tensor]:
        number_of_samples = len(self.scores)
        target_size = int(self.downsampling_ratio * number_of_samples / 100)
        selected_indices, weights = self._select_indexes_from_scores(target_size)
        selected_ids = [self.index_sampleid_map[index] for index in selected_indices]
        return selected_ids, weights

    def init_downsampler(self) -> None:
        self.scores = []
        self.index_sampleid_map = []

    def _select_indexes_from_scores(self, target_size: int) -> tuple[list[int], torch.Tensor]:
        return np.argsort(self.scores[::-1])[:target_size], torch.ones(target_size).float()
