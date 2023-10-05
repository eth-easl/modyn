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
    """
    Strategy introduced in:
    GRAD-MATCH:Gradient Matching based Data Subset Selection for Efficient Deep Model Training (Killamsetty et. al.)
    Implementation adapted from:
    DEEPCORE https://raw.githubusercontent.com/PatrickZH/DeepCore/main/deepcore/methods/gradmatch.py
    This strategy collects the Gradients (leveraging the abstract class AbstractMatrixDownsamplingStrategy) and then
    selects the samples using Orthogonal Matching Pursuit (OMP). The goal is to find a sparse vector x such that
    Ax = b, where A is the matrix containing the last layer gradients, and b is the mean across every dimension.
    Note that DEEPCORE proposes two versions, one requiring a validation dataset. Such a dataset is not available
    in modyn; thus, that version is unavailable. The vector b is the mean of the gradients of the validation dataset
    in this alternative version.
    """

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
            cur_weights = orthogonal_matching_pursuit_np(matrix.T, cur_val_gradients, budget=target_size)
        else:
            cur_weights = orthogonal_matching_pursuit(
                torch.Tensor(matrix).T, torch.Tensor(cur_val_gradients), budget=target_size
            )
        selection_result = np.nonzero(cur_weights)[0]
        weights = torch.tensor(cur_weights[selection_result])

        return selection_result.tolist(), weights
