import logging
from typing import Any, Optional, Union

import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)

logger = logging.getLogger(__name__)


class RemoteGradNormDownsampling(AbstractRemoteDownsamplingStrategy):
    """
    Method adapted from
    Not All Samples Are Created Equal: Deep Learning with Importance Sampling (Katharopoulos, Fleuret)
    The main idea is that the norm of the gradient up to the penultimate layer can be used to measure the importance
    of each sample. This is particularly convenient if the loss function is CrossEntropy since a closed form solution
    exists and thus no derivatives are needed (so it's marginally more expensive than computing the loss)
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
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)

        self.per_sample_loss_fct = per_sample_loss

        self.probabilities: list[torch.Tensor] = []
        self.number_of_points_seen = 0

    def init_downsampler(self) -> None:
        self.probabilities = []
        self.index_sampleid_map: list[int] = []
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: Union[dict[str, torch.Tensor], torch.Tensor],
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: Optional[torch.Tensor] = None,
    ) -> None:
        last_layer_gradients = self._compute_last_layer_gradient_wrt_loss_sum(
            self.per_sample_loss_fct, forward_output, target
        )
        if last_layer_gradients.dim() == 1:
            last_layer_gradients = last_layer_gradients.unsqueeze(1)
        scores = torch.linalg.vector_norm(last_layer_gradients, dim=1).cpu()
        self.probabilities.append(scores)
        self.number_of_points_seen += forward_output.shape[0]
        self.index_sampleid_map += sample_ids

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if len(self.probabilities) == 0:
            logger.warning("Empty probabilities, cannot select any points.")
            return [], torch.Tensor([])

        # select always at least 1 point
        target_size = max(int(self.downsampling_ratio * self.number_of_points_seen / self.ratio_max), 1)

        probabilities = torch.cat(self.probabilities, dim=0)
        # for whatever reason the probabilities may contain nan or inf, we replace them with 0
        probabilities[torch.isnan(probabilities)] = 0.0
        probabilities[torch.isinf(probabilities)] = 0.0
        sum_probability = probabilities.sum()
        if torch.isclose(sum_probability, torch.tensor(0.0)):
            logger.warning("Sum of probabilities is zero; Possibly all gradients are zero. Cannot normalize.")
            logger.warning("uniformly random select points")
            # random select without replacement
            downsampled_idxs = torch.randperm(probabilities.shape[0])[:target_size]
            weights = torch.ones(target_size, dtype=torch.float32, device=probabilities.device) * 1e-6
        else:
            probabilities = probabilities / sum_probability
            downsampled_idxs = torch.multinomial(probabilities, target_size, replacement=False)
            # lower probability, higher weight to reduce the variance
            # the weights are in two cases:
            # 1. if the probability is 0, the weight is 1e-6
            # 2. if the probability is not 0, the weight is 1 / (number_of_points_seen * probability)
            weights = torch.ones(target_size, dtype=torch.float32, device=probabilities.device) * 1e-6
            zero_idxs = torch.isclose(probabilities[downsampled_idxs], torch.tensor(0.0))
            non_zero_downsampled_idxs = downsampled_idxs[~zero_idxs]
            weights[~zero_idxs] = 1.0 / (self.number_of_points_seen * probabilities[non_zero_downsampled_idxs])
        # selected_ids = [self.index_sampleid_map[sample] for sample in downsampled_idxs]
        return downsampled_idxs.tolist(), weights

    @property
    def requires_grad(self) -> bool:
        if isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss):
            return False

        return True
