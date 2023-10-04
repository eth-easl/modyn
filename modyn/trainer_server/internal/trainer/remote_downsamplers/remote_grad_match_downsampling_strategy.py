from typing import Any

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_matrix_downsampling_strategy import (
    AbstractMatrixDownsamplingStrategy,
    MatrixContent,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.orthogonal_matching_pursuit import (
    orthogonal_matching_pursuit,
    orthogonal_matching_pursuit_np,
)


class RemoteGradMatchDownsamplingStrategy(AbstractMatrixDownsamplingStrategy):
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
        self.matrix_content = MatrixContent.GRADIENTS

    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        cur_val_gradients = np.mean(matrix, axis=0)

        if self.device == "cpu":
            # Compute OMP on numpy
            cur_weights = orthogonal_matching_pursuit_np(matrix.T, cur_val_gradients, budget=target_size)
        else:
            cur_weights = orthogonal_matching_pursuit(
                torch.Tensor(matrix).T, torch.Tensor(cur_val_gradients), budget=target_size
            )
        selection_result = np.nonzero(cur_weights)[0]
        weights = torch.tensor(cur_weights[selection_result])

        return selection_result.tolist(), weights
