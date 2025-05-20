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
    """
    Strategy introduced in:
    Active learning for convolutional neural networks: A coreset approach (Sener and Savarese)
    Implementation adapted from:
    DEEPCORE https://raw.githubusercontent.com/PatrickZH/DeepCore/main/deepcore/methods/kcentergreedy.py
    This strategy collects the Embeddings (leveraging the abstract class AbstractMatrixDownsamplingStrategy)
    and then selects the samples by clustering them in the embedding space. The clustering algorithm is k-center.
    This strategy was introduced first for active learning to discover which points are worth receiving a label.
    Hence, this class does not compute weights and returns a tensor of ones instead.
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
        super().__init__(
            pipeline_id,
            trigger_id,
            batch_size,
            params_from_selector,
            modyn_config,
            per_sample_loss,
            device,
            MatrixContent.EMBEDDINGS,
           
        )
        self.metric = euclidean_dist

    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        selected_indices = k_center_greedy(
            matrix, budget=target_size, metric=self.metric, device=self.device, print_freq=None
        )

        # no weights are returned by this technique
        selected_weights = torch.ones(len(selected_indices))

        return selected_indices, selected_weights
