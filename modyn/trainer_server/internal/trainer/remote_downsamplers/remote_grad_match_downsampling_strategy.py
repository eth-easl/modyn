# remote_grad_match_downsampling_strategy.py

import numpy as np
import torch
from typing import Any

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_matrix_downsampling_strategy import (
    AbstractMatrixDownsamplingStrategy,
    MatrixContent,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    FULL_GRAD_APPROXIMATION,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.orthogonal_matching_pursuit import (
    orthogonal_matching_pursuit,
    orthogonal_matching_pursuit_np,
)

class RemoteGradMatchDownsamplingStrategy(AbstractMatrixDownsamplingStrategy):
    """
    Now with an optional 'generative=True' for sequence-level usage.
    We'll treat each entire sequence as one sample. The matrix has shape (#samples, gradient_dim).
    Very expensive for big generative datasets, but here's the approach.
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
    ):
        self.generative = generative
        self.full_grad_approximation = params_from_selector["full_grad_approximation"]
        assert self.full_grad_approximation in FULL_GRAD_APPROXIMATION
        super().__init__(
            pipeline_id,
            trigger_id,
            batch_size,
            params_from_selector,
            modyn_config,
            per_sample_loss,
            device,
            (
                MatrixContent.LAST_LAYER_GRADIENTS
                if self.full_grad_approximation == "LastLayer"
                else MatrixContent.LAST_TWO_LAYERS_GRADIENTS
            ),
        )

    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        # matrix shape => (#samples, gradient_dim)
        # we compute the mean across #samples to approximate full gradient
        cur_val_gradients = np.mean(matrix, axis=0)  # shape(gradient_dim,)

        # Use OMP to find subset
        if self.device == "cpu":
            cur_weights = orthogonal_matching_pursuit_np(matrix.T, cur_val_gradients, budget=target_size)
        else:
            cur_weights = orthogonal_matching_pursuit(
                torch.Tensor(matrix).T, torch.Tensor(cur_val_gradients), budget=target_size
            )
        selection_result = np.nonzero(cur_weights)[0]  # indices of chosen
        weights = torch.tensor(cur_weights[selection_result])

        return selection_result.tolist(), weights
