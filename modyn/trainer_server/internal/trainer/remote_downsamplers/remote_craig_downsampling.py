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


class RemoteCraigDownsamplingStrategy(AbstractPerLabelRemoteDownsamplingStrategy):
    """
    Strategy introduced in:
    Data-efficient Training of Machine Learning Models
    Implementation adapted from:
    DEEPCORE https://raw.githubusercontent.com/PatrickZH/DeepCore/main/deepcore/methods/craig.py
    This strategy selects points via submodular maximization of a per-class FacilityLocation function. The score is
    proportional to the Euclidean distance between the two samples' gradients.
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
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        self.criterion = per_sample_loss

        self.selection_batch = params_from_selector["selection_batch"]
        self.greedy = params_from_selector["greedy"]
        self.full_grad_approximation = params_from_selector["full_grad_approximation"]
        assert self.full_grad_approximation in FULL_GRAD_APPROXIMATION

        if self.greedy not in OPTIMIZER_CHOICES:
            raise ValueError(
                f"The required Greedy optimizer is not available. Pick one of the following: {OPTIMIZER_CHOICES}"
            )

        self.requires_coreset_supporting_module = self.full_grad_approximation == "LastLayerWithEmbedding"
        self.requires_data_label_by_label = True

        # Samples are supplied label by label (this class is instance of AbstractPerLabelRemoteDownsamplingStrategy).
        # The following list keeps the gradients of the current label. When all the samples belonging to the current
        # label have been seen, the scores are computed and the list is emptied
        self.current_class_gradients: list[torch.Tensor] = []
        # distance_matrix[i,j] = 0 if label[i]!=label[j] else is proportional to the euclidean distance between the
        # two samples in the gradient space
        self.distance_matrix = np.zeros([0, 0])

        # if true, the downsampling is balanced across classes ex class sizes = [10, 50, 30] and 50% downsampling
        # yields the following downsampled class sizes [5, 25, 15] while without balance something like [0, 45, 0] can
        # happen
        self.balance = params_from_selector["balance"]
        if self.balance:
            self.already_selected_samples: list[int] = []
            self.already_selected_weights = torch.tensor([]).float()

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        # Slightly different implementation for BTS and STB since in STB points are supplied class by class while in
        # BTS are not. STB will always use the first branch, BTS will typically (might use the first if all the points
        # belong to the same class) use the second one
        # The problem with CRAIG is that the algorithm considers class when selecting points. This is not a problem in
        # STB since there is a dedicated dataloader to receive the samples as desired. In BTS it is different because
        # potentially in the same batch, there could be samples from different classes. If this happens, the algorithm
        # is used in a different way (receiving samples class by class, emulating what happens in STB).

        different_targets_in_this_batch = target.unique()
        if len(different_targets_in_this_batch) == 1:
            self._inform_samples_single_class(sample_ids, forward_output, target, embedding)
        else:
            for current_target in different_targets_in_this_batch:
                mask = target == current_target
                this_target_sample_ids = [sample_ids[i] for i, keep in enumerate(mask) if keep]
                sub_embedding = embedding[mask] if embedding is not None else None
                self._inform_samples_single_class(
                    this_target_sample_ids, forward_output[mask], target[mask], sub_embedding
                )
                self.inform_end_of_current_label()

    def _inform_samples_single_class(
        self,
        sample_ids: list[int],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None,
    ) -> None:
        forward_output, target = unsqueeze_dimensions_if_necessary(forward_output, target)
        if self.full_grad_approximation == "LastLayerWithEmbedding":
            assert embedding is not None
            grads_wrt_loss_sum = self._compute_last_two_layers_gradient_wrt_loss_sum(
                self.criterion, forward_output, target, embedding
            )
        else:
            grads_wrt_loss_sum = self._compute_last_layer_gradient_wrt_loss_sum(self.criterion, forward_output, target)
        batch_num = target.shape[0]
        grads_wrt_loss_mean = grads_wrt_loss_sum / batch_num
        self.current_class_gradients.append(grads_wrt_loss_mean.detach().cpu().numpy())
        # keep the mapping index<->sample_id
        self.index_sampleid_map += sample_ids

    def add_to_distance_matrix(self, submatrix: np.ndarray) -> None:
        # compute the new size of the matrix
        current_size = self.distance_matrix.shape[0]
        new_size = current_size + submatrix.shape[0]

        # copy the old matrix into the new one
        new_matrix = np.zeros([new_size, new_size])
        new_matrix[:current_size, :current_size] = self.distance_matrix

        # add the new submatrix
        new_matrix[
            current_size:,
            current_size:,
        ] = submatrix

        self.distance_matrix = new_matrix

    def inform_end_of_current_label(self) -> None:
        if len(self.current_class_gradients) == 0:
            # no new gradients, just return
            return
        # compute the scores for each pair of samples belonging to the current class
        gradients = np.concatenate(self.current_class_gradients)
        matrix = -1.0 * euclidean_dist_pair_np(gradients)
        matrix -= np.min(matrix) - 1e-3
        # store the result in the matrix
        self.add_to_distance_matrix(matrix)
        # empty the gradients list
        self.current_class_gradients = []

        if self.balance:
            # here we select the points if we want to keep a balance across classes
            this_class_samples, this_class_weights = self._select_points_from_distance_matrix()
            self.already_selected_samples += this_class_samples
            self.already_selected_weights = torch.cat((self.already_selected_weights, this_class_weights))
            self.distance_matrix = np.zeros([0, 0])
            self.index_sampleid_map: list[int] = []

    def calc_weights(self, matrix: np.ndarray, result: np.ndarray) -> torch.Tensor:
        min_sample = np.argmax(matrix[result], axis=0)
        weights = np.ones(np.sum(result) if result.dtype == bool else len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return torch.tensor(weights).float()

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if self.balance:
            # return a shuffled version, otherwise training happens class by class
            return _shuffle_list_and_tensor(self.already_selected_samples, self.already_selected_weights)

        if len(self.current_class_gradients) != 0:
            # conclude the last class if there are still samples
            self.inform_end_of_current_label()
        return self._select_points_from_distance_matrix()

    def _select_points_from_distance_matrix(self) -> tuple[list[int], torch.Tensor]:
        number_of_samples = self.distance_matrix.shape[0]
        target_size = max(int(self.downsampling_ratio * number_of_samples / self.ratio_max), 1)

        all_index = np.arange(number_of_samples)
        submod_function = FacilityLocation(index=all_index, similarity_matrix=self.distance_matrix)
        submod_optimizer = submodular_optimizer.__dict__[self.greedy](
            args=Namespace(print_freq=None), index=all_index, budget=target_size
        )
        selection_result = submod_optimizer.select(
            gain_function=submod_function.calc_gain_batch,
            update_state=submod_function.update_state,
            batch=self.selection_batch,
        )
        weights = self.calc_weights(self.distance_matrix, selection_result)
        selected_ids = [self.index_sampleid_map[sample] for sample in selection_result]
        return selected_ids, weights

    def init_downsampler(self) -> None:
        self.index_sampleid_map = []
        self.current_class_gradients = []
        self.distance_matrix = np.zeros([0, 0])
        if self.balance:
            self.already_selected_samples = []
            self.already_selected_weights = torch.tensor([]).float()

    @property
    def requires_grad(self) -> bool:
        return True
