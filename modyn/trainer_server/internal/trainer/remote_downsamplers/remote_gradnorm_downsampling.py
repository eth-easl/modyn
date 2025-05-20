import logging
from typing import Any

import torch
import torch.nn
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
    unsqueeze_dimensions_if_necessary,
)

logger = logging.getLogger(__name__)


class RemoteGradNormDownsampling(AbstractRemoteDownsamplingStrategy):
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
    ) -> None:
        super().__init__(pipeline_id, trigger_id, batch_size, params_from_selector, modyn_config, device)
        self.per_sample_loss_fct = per_sample_loss
        self.generative = generative
        self.probabilities: list[torch.Tensor] = []
        self.index_sampleid_map: list[int] = []
        self.number_of_points_seen = 0

    def init_downsampler(self) -> None:
        self.probabilities.clear()
        self.index_sampleid_map.clear()
        self.number_of_points_seen = 0

    def inform_samples(
        self,
        sample_ids: list[int],
        forward_input: dict[str, torch.Tensor] | torch.Tensor,
        forward_output: torch.Tensor,
        target: torch.Tensor,
        embedding: torch.Tensor | None = None,
    ) -> None:
        # expand dims if needed
        forward_output, target = unsqueeze_dimensions_if_necessary(forward_output, target)

        # basic shape checks
        assert forward_output.dim() in (2, 3), "forward_output must be 2D or 3D"
        B = forward_output.size(0)

        # ensure target is integer type on same device
        assert target.dtype in (torch.int64, torch.long), "target must be long"
        assert target.device == forward_output.device, "target must live on same device"

        # check label ranges
        V = forward_output.shape[-1]
        t_flat = target.view(-1)
        assert (t_flat >= -100).all(), "found target < -100"
        # allow ignore_index = -100
        valid = (t_flat >= 0) & (t_flat < V)
        mask = t_flat == -100
        assert (valid | mask).all(), "found target outside [0, V-1] or != ignore_index"

        # choose scoring path
        if isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss) and not self.generative:
            per_sample = self.per_sample_loss_fct(forward_output, target)
            scores = per_sample.detach().cpu()
        else:
            if self.generative:
                # flatten sequences
                T = forward_output.size(1)
                fo_flat = forward_output.reshape(B * T, V)
                tgt_flat = target.reshape(B * T)

                # recompute with asserts
                grads = self._compute_last_layer_gradient_wrt_loss_sum(
                    self.per_sample_loss_fct, fo_flat, tgt_flat
                )
                # grads should be Tensor[BT, D]
                assert isinstance(grads, torch.Tensor), "gradients returned None"
                assert grads.size(0) == B * T, "unexpected grad batch size"
                print(grads)
                print(grads.size())
                # reshape and score
                D = grads.size(-1)
                grads = grads.view(B, T, D)
                norms = torch.linalg.vector_norm(grads, dim=2)  # (B, T)
                scores = norms.sum(dim=1).cpu()

            else:
                grads = self._compute_last_layer_gradient_wrt_loss_sum(
                    self.per_sample_loss_fct, forward_output, target
                )
                assert isinstance(grads, torch.Tensor), "gradients returned None"
                assert grads.size(0) == B, "unexpected grad batch size"
                scores = torch.linalg.vector_norm(grads, dim=1).cpu()

        # final safety
        assert torch.isfinite(scores).all(), "scores contain nan or inf"

        self.probabilities.append(scores)
        self.number_of_points_seen += B
        self.index_sampleid_map.extend(sample_ids)

    def select_points(self) -> tuple[list[int], torch.Tensor]:
        if not self.probabilities:
            logger.warning("Empty probabilities, cannot select any points.")
            return [], torch.Tensor([])

        total = self.downsampling_ratio * self.number_of_points_seen / self.ratio_max
        target_size = max(int(total), 1)

        probs = torch.cat(self.probabilities, dim=0)
        probs = probs / probs.sum()

        idxs = torch.multinomial(probs, target_size, replacement=False)
        weights = 1.0 / (self.number_of_points_seen * probs[idxs])

        selected_ids = [self.index_sampleid_map[i] for i in idxs.tolist()]
        return selected_ids, weights

    @property
    def requires_grad(self) -> bool:
        # we only need true gradients if not using standard CE
        return not isinstance(self.per_sample_loss_fct, torch.nn.CrossEntropyLoss)
