import logging
import random
from typing import Any

import torch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)


class RemoteRS2Downsampling(AbstractRemoteDownsamplingStrategy):
    """
    Method adapted from REPEATED RANDOM SAMPLING FOR MINIMIZING THE TIME-TO-ACCURACY OF LEARNING (Okanovic+, 2024)
    https://openreview.net/pdf?id=JnRStoIuTe
    """

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        batch_size: int,
        params_from_selector: dict,
        modyn_config: dict,
        # pylint: disable-next=unused-argument
        per_sample_loss: Any,
        device: str,
        generative: bool = False
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)
        self.forward_required = False
        self.supports_bts = False
        self._all_sample_ids: list[int] = []
        self._subsets: list[list[int]] = []
        self._current_subset = -1
        self._with_replacement: bool = params_from_selector["replacement"]
        self._first_epoch = True

    def init_downsampler(self) -> None:
        pass  # We take care of that in inform_samples

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        # We only need to collect the sample information once
        if self._first_epoch:
            self._all_sample_ids.extend(sample_ids)

    def _epoch_step_wr(self, target_size: int) -> None:
        random.shuffle(self._all_sample_ids)
        self._subsets = [self._all_sample_ids[:target_size]]
        self._current_subset = 0

    def _epoch_step_no_r(self, target_size: int) -> None:
        if (max_subset := len(self._all_sample_ids) // target_size) == 0:
            self._current_subset = 0
            self._subsets = [[]]
            return

        self._current_subset += 1

        # len(self._subsets) == 0 holds in the very first epoch
        if self._current_subset >= max_subset or len(self._subsets) == 0:
            random.shuffle(self._all_sample_ids)
            self._current_subset = 0
            self._subsets = [self._all_sample_ids[i * target_size : (i + 1) * target_size] for i in range(max_subset)]

    def _epoch_step(self) -> None:
        target_size = max(int(self.downsampling_ratio * len(self._all_sample_ids) / self.ratio_max), 1)

        if self._with_replacement:
            self._epoch_step_wr(target_size)
        else:
            self._epoch_step_no_r(target_size)

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        self._first_epoch = False
        self._epoch_step()
        assert self._current_subset < len(
            self._subsets
        ), f"Inconsistent state: {self._current_subset}\n{self._subsets}\n{self._first_epoch}\n{self._all_sample_ids}"
        return self._subsets[self._current_subset], torch.ones(len(self._subsets[self._current_subset]))

    @property
    def requires_grad(self) -> bool:
        return False
