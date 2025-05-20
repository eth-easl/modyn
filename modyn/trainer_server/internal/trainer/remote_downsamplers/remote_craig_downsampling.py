import logging
from argparse import Namespace
from typing import Any

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_per_label_remote_downsample_strategy import (
    AbstractPerLabelRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    FULL_GRAD_APPROXIMATION,
    unsqueeze_dimensions_if_necessary,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils import submodular_optimizer
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.euclidean import euclidean_dist_pair_np
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.shuffling import _shuffle_list_and_tensor
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.submodular_function import (
    FacilityLocation,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.submodular_optimizer import (
    OPTIMIZER_CHOICES,
)

logger = logging.getLogger(__name__)


class RemoteCraigDownsamplingStrategy(AbstractPerLabelRemoteDownsamplingStrategy):
    """
    Adapted to handle generative tasks at sequence-level.
    If generative=True, forward_output is shape (B, T, V).
    We'll compute per-sample gradient means by summing across tokens in the cross-entropy.
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
        self.criterion = per_sample_loss
        self.generative = generative

        self.selection_batch = params_from_selector.get("selection_batch")
        self.greedy = params_from_selector.get("greedy")
        self.full_grad_approximation = params_from_selector.get("full_grad_approximation")
        assert self.full_grad_approximation in FULL_GRAD_APPROXIMATION

        if self.greedy not in OPTIMIZER_CHOICES:
            raise ValueError(f"Greedy optimizer not in {OPTIMIZER_CHOICES}")

        self.requires_coreset_supporting_module = (self.full_grad_approximation == "LastLayerWithEmbedding")
        self.requires_data_label_by_label = True

        self.current_class_gradients: list[torch.Tensor] = []
        self.distance_matrix = np.zeros((0, 0))
        self.index_sampleid_map: list[int] = []

        self.balance = params_from_selector.get("balance", False)
        if self.balance:
            self.already_selected_samples: list[int] = []
            self.already_selected_weights = torch.tensor([]).float()

    def init_downsampler(self) -> None:
        self.index_sampleid_map.clear()
        self.current_class_gradients.clear()
        self.distance_matrix = np.zeros((0, 0))
        if self.balance:
            self.already_selected_samples.clear()
            self.already_selected_weights = torch.tensor([]).float()

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        # handle generative or classification shapes
        forward_output, target = unsqueeze_dimensions_if_necessary(forward_output, target)
        # separate by labels if needed
        unique = target.unique()
        if len(unique) == 1:
            self._inform_samples_single_class(sample_ids, forward_output, target, embedding)
        else:
            for t in unique:
                mask = (target == t)
                ids = [sid for sid, m in zip(sample_ids, mask) if m]
                fo_sub = forward_output[mask]
                tgt_sub = target[mask]
                emb_sub = embedding[mask] if embedding is not None else None
                self._inform_samples_single_class(ids, fo_sub, tgt_sub, emb_sub)
                self.inform_end_of_current_label()

    def _inform_samples_single_class(
        self,
        sample_ids: list[int],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None,
    ) -> None:
        forward_output, target = unsqueeze_dimensions_if_necessary(forward_output, target)
        B = forward_output.size(0)
        # flatten sequence dimension if generative
        if self.generative:
            # forward_output: (B, T, V) -> (B*T, V)
            T, V = forward_output.shape[1], forward_output.shape[2]
            fo_flat = forward_output.reshape(B * T, V)
            tgt_flat = target.reshape(B * T)
            grads = (self._compute_last_two_layers_gradient_wrt_loss_sum
                     if self.full_grad_approximation == "LastLayerWithEmbedding"
                     else self._compute_last_layer_gradient_wrt_loss_sum)(
                         self.criterion, fo_flat, tgt_flat,
                         embedding.reshape(B * T, -1) if self.full_grad_approximation == "LastLayerWithEmbedding" else None
                     )
            # reshape back: (B*T, D) -> (B, T, D) then mean over T
            D = grads.shape[1]
            grads = grads.view(B, T, D).mean(dim=1)
        else:
            # classification path
            grads = (self._compute_last_two_layers_gradient_wrt_loss_sum
                     if self.full_grad_approximation == "LastLayerWithEmbedding"
                     else self._compute_last_layer_gradient_wrt_loss_sum)(
                         self.criterion, forward_output, target,
                         embedding if self.full_grad_approximation == "LastLayerWithEmbedding" else None
                     )
        # detach and move to CPU
        self.current_class_gradients.append(grads.detach().cpu().numpy())
        self.index_sampleid_map.extend(sample_ids)

    def inform_end_of_current_label(self) -> None:
        if not self.current_class_gradients:
            return
        grads = np.concatenate(self.current_class_gradients, axis=0)
        sim = -euclidean_dist_pair_np(grads)
        sim -= sim.min() - 1e-3
        self._add_to_distance_matrix(sim)
        self.current_class_gradients.clear()
        if self.balance:
            sel, w = self._select_points_from_distance_matrix()
            self.already_selected_samples.extend(sel)
            self.already_selected_weights = torch.cat((self.already_selected_weights, w))
            self.distance_matrix = np.zeros((0, 0))
            self.index_sampleid_map.clear()

    def _add_to_distance_matrix(self, submatrix: np.ndarray) -> None:
        n0 = self.distance_matrix.shape[0]
        n1 = submatrix.shape[0]
        new = np.zeros((n0 + n1, n0 + n1))
        new[:n0, :n0] = self.distance_matrix
        new[n0:, n0:] = submatrix
        self.distance_matrix = new

    def calc_weights(self, matrix: np.ndarray, result: np.ndarray) -> torch.Tensor:
        mins = np.argmax(matrix[result], axis=0)
        weights = np.ones(result.sum() if result.dtype == bool else len(result))
        for i in mins:
            weights[i] += 1
        return torch.tensor(weights).float()

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if self.balance:
            return _shuffle_list_and_tensor(self.already_selected_samples, self.already_selected_weights)
        if self.current_class_gradients:
            self.inform_end_of_current_label()
        return self._select_points_from_distance_matrix()

    def _select_points_from_distance_matrix(self) -> tuple[list[int], torch.Tensor]:
        N = self.distance_matrix.shape[0]
        k = max(int(self.downsampling_ratio * N / self.ratio_max), 1)
        idxs = np.arange(N)
        func = FacilityLocation(index=idxs, similarity_matrix=self.distance_matrix)
        opt = submodular_optimizer.__dict__[self.greedy](
            args=Namespace(print_freq=None), index=idxs, budget=k
        )
        res = opt.select(gain_function=func.calc_gain_batch, update_state=func.update_state, batch=self.selection_batch)
        w = self.calc_weights(self.distance_matrix, res)
        sel = [self.index_sampleid_map[i] for i in res]
        return sel, w

    @property
    def requires_grad(self) -> bool:
        return True  
