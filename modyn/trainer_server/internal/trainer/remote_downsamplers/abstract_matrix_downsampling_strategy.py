from abc import abstractmethod
from enum import Enum
from typing import Any, Optional, Union

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_per_label_remote_downsample_strategy import (
    AbstractPerLabelRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.shuffling import _shuffle_list_and_tensor

MatrixContent = Enum("MatrixContent", ["EMBEDDINGS", "GRADIENTS"])


class AbstractMatrixDownsamplingStrategy(AbstractPerLabelRemoteDownsamplingStrategy):
    """
    Class to abstract the common behaviour of many downsampling strategies that collect the gradients or the embeddings
    (thus a Matrix) and then select the points based on some method-specific metric (submodular, clustering, OMP...).
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
        matrix_content: MatrixContent,
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        self.criterion = per_sample_loss

        # This class uses the embedding recorder
        self.requires_coreset_supporting_module = True
        self.matrix_elements: list[torch.Tensor] = []

        # actual classes must specify which content should be stored. Can be either Gradients or Embeddings. Use the
        # enum defined above to specify what should be stored
        self.matrix_content = matrix_content

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
        forward_input: Union[dict[str, torch.Tensor], torch.Tensor],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        assert embedding is not None
        assert self.matrix_content is not None

        if self.matrix_content == MatrixContent.GRADIENTS:
            new_elements = self._compute_gradients(forward_output, target, embedding)
        elif self.matrix_content == MatrixContent.EMBEDDINGS:
            new_elements = embedding.detach().cpu()
        else:
            raise AssertionError("The required content does not exits.")

        self.matrix_elements.append(new_elements)
        # keep the mapping index<->sample_id
        self.index_sampleid_map += sample_ids

    def _compute_gradients(
        self, forward_output: torch.Tensor, target: torch.Tensor, embedding: torch.Tensor
    ) -> torch.Tensor:
        loss = self.criterion(forward_output, target).mean()
        embedding_dim = embedding.shape[1]
        num_classes = forward_output.shape[1]
        batch_num = target.shape[0]
        # compute the gradient for each element provided
        with torch.no_grad():
            bias_parameters_grads = torch.autograd.grad(loss, forward_output)[0]
            weight_parameters_grads = embedding.view(batch_num, 1, embedding_dim).repeat(
                1, num_classes, 1
            ) * bias_parameters_grads.view(batch_num, num_classes, 1).repeat(1, 1, embedding_dim)
            gradients = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu().numpy()
        return gradients

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
        return self.matrix_content == MatrixContent.GRADIENTS
