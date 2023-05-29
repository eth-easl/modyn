from typing import Any

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampler_temporary_storage import (
    AbstractRemoteDownsamplingTemporaryStorage,
)


class RemoteLossDownsampling(AbstractRemoteDownsamplingTemporaryStorage):
    def __init__(
        self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict, per_sample_loss: Any
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector)

        self.per_sample_loss_fct = per_sample_loss

    def get_scores(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = self.per_sample_loss_fct(forward_output, target).detach()
        return scores

    def accumulate_sample_then_batch(
        self,
        model_output: torch.Tensor,
        target: torch.Tensor,
        sample_ids: list,
    ) -> None:
        scores = self.get_scores(model_output, target).numpy()
        self.sample_then_batch_temporary_storage.accumulate(sample_ids, scores)
