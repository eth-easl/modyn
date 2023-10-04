from typing import Any

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_matrix_downsampling_strategy import (
    AbstractMatrixDownsamplingStrategy,
    MatrixContent,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.euclidean import euclidean_dist
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.k_center_greedy import k_center_greedy


class RemoteKcenterGreedyDownsamplingStrategy(AbstractMatrixDownsamplingStrategy):
    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        per_sample_loss: Any,
        device: str,
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, per_sample_loss, device)

        self.matrix_content = MatrixContent.EMBEDDINGS
        self.metric = euclidean_dist

    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        selected_indices = k_center_greedy(
            matrix, budget=target_size, metric=self.metric, device=self.device, print_freq=None
        )

        # no weights are returned by this technique
        selected_weights = torch.ones((len(selected_indices)))

        return selected_indices, selected_weights
