from abc import abstractmethod
from enum import Enum
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


class MatrixContent(Enum):
    EMBEDDINGS = 1
    LAST_LAYER_GRADIENTS = 2
    LAST_TWO_LAYERS_GRADIENTS = 3


class AbstractMatrixDownsamplingStrategy(AbstractPerLabelRemoteDownsamplingStrategy):
    """Class to abstract the common behaviour of many downsampling strategies
    that collect the gradients or the embeddings (thus a Matrix) and then
    select the points based on some method-specific metric (submodular,
    clustering, OMP...)."""

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        per_sample_loss: Any,
        device: str,
        matrix_content: MatrixContent,
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        self.criterion = per_sample_loss

        self.matrix_elements: list[torch.Tensor] = []

        # actual classes must specify which content should be stored. Can be either Gradients or Embeddings. Use the
        # enum defined above to specify what should be stored
        self.matrix_content = matrix_content

        self.requires_coreset_supporting_module = self.matrix_content in [
            MatrixContent.LAST_TWO_LAYERS_GRADIENTS,
            MatrixContent.EMBEDDINGS,
        ]

        # if true, the downsampling is balanced across classes ex class sizes = [10, 50, 30] and 50% downsampling
        # yields the following downsampled class sizes [5, 25, 15] while without balance something like [0, 45, 0] can
        # happen
        self.balance = params_from_selector["balance"]
        if self.balance:
            # Selection happens class by class. These data structures are used to store the selection results
            self.already_selected_samples: list[int] = []
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
        batch_size = len(sample_ids)
        assert self.matrix_content is not None
        forward_output, target = unsqueeze_dimensions_if_necessary(forward_output, target)
        if self.matrix_content == MatrixContent.LAST_LAYER_GRADIENTS:
            grads_wrt_loss_sum = self._compute_last_layer_gradient_wrt_loss_sum(self.criterion, forward_output, target)
            grads_wrt_loss_mean = grads_wrt_loss_sum / batch_size
            new_elements = grads_wrt_loss_mean.detach().cpu()
        elif self.matrix_content == MatrixContent.LAST_TWO_LAYERS_GRADIENTS:
            assert embedding is not None
            # using the gradients w.r.t. the sum of the loss or the mean of the loss does not make a difference
            # since the scaling factor is the same for all samples. We use mean here to pass the unit test
            # containing the hard-coded values from deepcore
            grads_wrt_loss_sum = self._compute_last_two_layers_gradient_wrt_loss_sum(
                self.criterion, forward_output, target, embedding
            )
            grads_wrt_loss_mean = grads_wrt_loss_sum / batch_size
            new_elements = grads_wrt_loss_mean.detach().cpu()
        elif self.matrix_content == MatrixContent.EMBEDDINGS:
            assert embedding is not None
            new_elements = embedding.detach().cpu()
        else:
            raise AssertionError("The required content does not exits.")

        self.matrix_elements.append(new_elements)
        # keep the mapping index<->sample_id
        self.index_sampleid_map += sample_ids

    def inform_end_of_current_label(self) -> None:
        assert self.balance
        selected_samples, selected_weights = self._select_from_matrix()
        self.already_selected_samples += selected_samples
        self.already_selected_weights = torch.cat((self.already_selected_weights, selected_weights))
        self.matrix_elements = []
        self.index_sampleid_map: list[int] = []

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if self.balance:
            ids, weights = self.already_selected_samples, self.already_selected_weights
        else:
            ids, weights = self._select_from_matrix()

        return _shuffle_list_and_tensor(ids, weights)

    def _select_from_matrix(self) -> tuple[list[int], torch.Tensor]:
        matrix = np.concatenate(self.matrix_elements)
        number_of_samples = len(matrix)
        target_size = max(int(self.downsampling_ratio * number_of_samples / self.ratio_max), 1)
        selected_indices, weights = self._select_indexes_from_matrix(matrix, target_size)
        selected_ids = [self.index_sampleid_map[index] for index in selected_indices]
        return selected_ids, weights

    def init_downsampler(self) -> None:
        self.matrix_elements = []
        self.index_sampleid_map = []

    @abstractmethod
    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        # Here is where the actual selection happens
        raise NotImplementedError()

    @property
    def requires_grad(self) -> bool:
        # Default to true if None
        return self.matrix_content in [MatrixContent.LAST_LAYER_GRADIENTS, MatrixContent.LAST_TWO_LAYERS_GRADIENTS]
