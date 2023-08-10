from argparse import Namespace
from typing import Any

import numpy as np
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_matrix_downsampling_strategy import (
    AbstractMatrixDownsamplingStrategy,
    MatrixContent,
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
    def __init__(
        self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict, per_sample_loss: Any
    ):
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, per_sample_loss)
        self.matrix_content = MatrixContent.GRADIENTS

        # these arguments are required by DeepCore. Default values are used if not provided
        self.args = Namespace(**params_from_selector.get("deepcore_args", {}))
        if "print_freq" not in self.args:
            self.args.print_freq = None  # avoid printing when running the submodular optimization
        if "selection_batch" not in self.args:
            self.args.selection_batch = 64

        if "submodular_function" not in params_from_selector:
            raise ValueError(
                f"Please specify the submodular function used to select the datapoints. "
                f"Available functions: {SUBMODULAR_FUNCTIONS}, param submodular_function"
            )
        self._function = params_from_selector["submodular_function"]
        if self._function not in SUBMODULAR_FUNCTIONS:
            raise ValueError(
                f"The specified submodular function is not available. Pick one from {SUBMODULAR_FUNCTIONS}"
            )

        self._greedy = params_from_selector.get("greedy", "NaiveGreedy")
        if self._greedy not in OPTIMIZER_CHOICES:
            raise ValueError(
                f"The required Greedy optimizer is not available. Pick one of the following: {OPTIMIZER_CHOICES}"
            )

    def _select_indexes_from_matrix(self, matrix: np.ndarray, target_size: int) -> tuple[list[int], torch.Tensor]:
        number_of_samples = len(matrix)
        all_index = np.arange(number_of_samples)
        submod_function = submodular_function.__dict__[self._function](
            index=all_index, similarity_kernel=lambda a, b: cossim_np(matrix[a], matrix[b])
        )
        submod_optimizer = submodular_optimizer.__dict__[self._greedy](
            args=self.args, index=all_index, budget=target_size
        )
        selection_result = submod_optimizer.select(
            gain_function=submod_function.calc_gain, update_state=submod_function.update_state
        )

        # no weights are computed with this strategy
        weights = torch.ones(len(selection_result))
        return selection_result, weights
