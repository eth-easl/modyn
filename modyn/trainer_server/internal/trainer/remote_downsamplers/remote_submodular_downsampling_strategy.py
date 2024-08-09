from argparse import Namespace
from typing import Any

import numpy as np
import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_matrix_downsampling_strategy import (
    AbstractMatrixDownsamplingStrategy,
    MatrixContent,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    FULL_GRAD_APPROXIMATION,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils import (
    submodular_function,
    submodular_optimizer,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.cossim import cossim_np
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.submodular_function import (
    SUBMODULAR_FUNCTIONS,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.deepcore_utils.submodular_optimizer import (
    OPTIMIZER_CHOICES,
)


class RemoteSubmodularDownsamplingStrategy(AbstractMatrixDownsamplingStrategy):
    """
    Strategy introduced in:
    Submodular combinatorial information measures with applications in machine learning (Iyer et al.)
    Implementation adapted from:
    DEEPCORE https://raw.githubusercontent.com/PatrickZH/DeepCore/main/deepcore/methods/submodular.py
    This strategy collects the last layer gradients (leveraging the class AbstractMatrixDownsamplingStrategy)
    and then selects the samples by mapping the problem to submodular optimization. The user can select the submod
    function (available: FacilityLocation, GraphCut and LogDeterminant) and the optimizer to solve it greedily.
    The similarity between points is measured using the scaled and shifted cosine similarity between the last layer
    gradients. The shift is needed since some submodular functions (e.g. GraphCut) require non-negative values.
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

        self.selection_batch = params_from_selector["selection_batch"]

        if "submodular_function" not in params_from_selector:
            raise ValueError(
                f"Please specify the submodular function used to select the datapoints. "
                f"Available functions: {SUBMODULAR_FUNCTIONS}."
                f"Use the parameter param submodular_function"
            )
        self._function = params_from_selector["submodular_function"]
        if self._function not in SUBMODULAR_FUNCTIONS:
            raise ValueError(
                f"The specified submodular function is not available. "
                f"Pick one from {SUBMODULAR_FUNCTIONS}"
                f"Use the parameter param submodular_function"
            )

        self._greedy = params_from_selector.get("submodular_optimizer", "NaiveGreedy")
        if self._greedy not in OPTIMIZER_CHOICES:
            raise ValueError(
                f"The required Greedy optimizer is not available. "
                f"Pick one of the following: {OPTIMIZER_CHOICES}"
                f"Use the parameter param submodular_optimizer"
            )

    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        number_of_samples = len(matrix)
        all_index = np.arange(number_of_samples)
        submod_function = submodular_function.__dict__[self._function](
            index=all_index, similarity_kernel=lambda a, b: cossim_np(matrix[a], matrix[b])
        )
        submod_optimizer = submodular_optimizer.__dict__[self._greedy](
            args=Namespace(print_freq=None), index=all_index, budget=target_size
        )
        selection_result = submod_optimizer.select(
            gain_function=submod_function.calc_gain,
            update_state=submod_function.update_state,
            batch=self.selection_batch,
        )

        # no weights are computed with this strategy
        weights = torch.ones(len(selection_result))
        return selection_result, weights
