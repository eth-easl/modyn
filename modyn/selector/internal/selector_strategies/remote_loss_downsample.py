from typing import Any, Tuple, Union

import torch


class RemoteLossDownsampler:
    def __init__(self, model: Any, downsampled_batch_size: int, per_sample_loss_fct: Any) -> None:
        self.model = model
        self.downsampled_batch_size = downsampled_batch_size
        self.per_sample_loss_fct = per_sample_loss_fct

    def sample(
        self, data: Union[torch.Tensor, dict], target: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, dict], torch.Tensor, torch.Tensor]:
        output = self.model(data)
        scores = self.per_sample_loss_fct(output, target).detach()

        # sample according the score distribution
        probabilities = scores / scores.sum()

        downsampled_idxs = torch.multinomial(probabilities, self.downsampled_batch_size, replacement=True)

        weights = 1.0 / (len(target) * probabilities[downsampled_idxs])

        return data[downsampled_idxs], weights, target[downsampled_idxs]
