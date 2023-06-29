from typing import Any

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


class RemoteLossDownsampling(AbstractRemoteDownsamplingStrategy):
    def __init__(
        self, pipeline_id: int, trigger_id: int, batch_size: int, params_from_selector: dict, per_sample_loss: Any
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector)

        self.per_sample_loss_fct = per_sample_loss
        self.probabilities: list[torch.Tensor] = []
        self.number_of_points_seen = 0

    def get_scores(self, forward_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scores = self.per_sample_loss_fct(forward_output, target).detach()
        return scores

    def init_downsampler(self) -> None:
        self.probabilities = []
        self.index_sampleid_map: list[int] = []
        self.number_of_points_seen = 0

    def inform_samples(self, sample_ids: list[int], forward_output: torch.Tensor, target: torch.Tensor) -> None:
        scores = self.get_scores(forward_output, target)
        self.probabilities.append(scores)
        self.number_of_points_seen += forward_output.shape[0]
        self.index_sampleid_map += sample_ids

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        target_size = int(self.downsampling_ratio * self.number_of_points_seen / 100)

        probabilities = torch.cat(self.probabilities, dim=0)
        probabilities = probabilities / probabilities.sum()

        downsampled_idxs = torch.multinomial(probabilities, target_size, replacement=self.replacement)

        # lower probability, higher weight to reduce the variance
        weights = 1.0 / (self.number_of_points_seen * probabilities[downsampled_idxs])

        selected_ids = [self.index_sampleid_map[sample] for sample in downsampled_idxs]
        return selected_ids, weights
