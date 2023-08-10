from abc import abstractmethod
from enum import Enum
from typing import Any, Optional

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

MatrixContent = Enum("MatrixContent", ["EMBEDDINGS", "GRADIENTS"])


class AbstractMatrixDownsamplingStrategy(AbstractRemoteDownsamplingStrategy):
    def __init__(
        self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict, per_sample_loss: Any
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector)

        self.criterion = per_sample_loss

        # This class uses the embedding recorder
        self.requires_coreset_methods_support = True
        self.matrix_elements: list[torch.Tensor] = []

        self.matrix_content: Optional[MatrixContent] = None

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        assert embedding is not None
        assert self.matrix_content is not None

        if self.matrix_content == MatrixContent.GRADIENTS:
            to_be_added = self._compute_gradients(forward_output, target, embedding)
        elif self.matrix_content == MatrixContent.EMBEDDINGS:
            to_be_added = embedding.detach()
        else:
            raise AssertionError("The required content does not exits.")

        self.matrix_elements.append(to_be_added)
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

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        matrix = np.concatenate(self.matrix_elements)
        number_of_samples = len(matrix)
        target_size = int(self.downsampling_ratio * number_of_samples / 100)

        selected_indices, weights = self._select_indexes_from_matrix(matrix, target_size)
        selected_ids = [self.index_sampleid_map[index] for index in selected_indices]

        return selected_ids, weights

    def init_downsampler(self) -> None:
        self.matrix_elements = []

    @abstractmethod
    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        raise NotImplementedError()
