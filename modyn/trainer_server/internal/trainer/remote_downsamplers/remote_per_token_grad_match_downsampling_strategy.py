# remote_token_grad_match_downsampling_strategy.py
import logging
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

logger = logging.getLogger(__name__)


class RemoteTokenGradMatchDownsamplingStrategy(AbstractMatrixDownsamplingStrategy):
    """
    Token-level GradMatch for generative tasks (extremely expensive).
    Each token -> separate gradient vector. The matrix can get huge quickly.
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
    ):
        super().__init__(
            pipeline_id,
            trigger_id,
            batch_size,
            params_from_selector,
            modyn_config,
            per_sample_loss,
            device,
            MatrixContent.LAST_LAYER_GRADIENTS,  # or last_two_layers, but watch out
        )
        self.generative: bool = True
        logger.warning("Token-level GradMatch is extremely large overhead.")

    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        # matrix shape => (#tokens, gradient_dim)
        # we approximate full gradient as the average
        full_gradient = np.mean(matrix, axis=0)  # shape(gradient_dim,)

        # OMP
        if self.device == "cpu":
            cur_weights = orthogonal_matching_pursuit_np(matrix.T, full_gradient, budget=target_size)
        else:
            cur_weights = orthogonal_matching_pursuit(
                torch.Tensor(matrix).T, torch.Tensor(full_gradient), budget=target_size
            )
        selection_result = np.nonzero(cur_weights)[0]
        weights = torch.tensor(cur_weights[selection_result])
        return selection_result.tolist(), weights
