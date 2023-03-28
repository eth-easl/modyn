from typing import Tuple, Union

import torch


class AbstractRemoteDownsamplingStrategy:
    def __init__(self, model: torch.nn, params_from_selector: dict) -> None:
        self.model = model

        assert "downsampled_batch_size" in params_from_selector
        self.downsampled_batch_size = params_from_selector["downsampled_batch_size"]

        self.replacement = params_from_selector.get("replacement", True)

    def sample(
        self, data: Union[torch.Tensor, dict], target: torch.Tensor, sample_ids: list
    ) -> Tuple[Union[torch.Tensor, dict], torch.Tensor, torch.Tensor, list, torch.Tensor]:
        probabilities, forward_outputs = self.get_probabilities_and_forward_outputs(data, target)
        downsampled_idxs = torch.multinomial(probabilities, self.downsampled_batch_size, replacement=self.replacement)

        weights = 1.0 / (len(sample_ids) * probabilities[downsampled_idxs])

        selected_sample_ids = [sample_ids[i] for i in downsampled_idxs]

        if isinstance(data, torch.Tensor):
            data = data[downsampled_idxs]
        elif isinstance(data, dict):
            data = {key: tensor[downsampled_idxs] for key, tensor in data.items()}

        return data, weights, target[downsampled_idxs], selected_sample_ids, forward_outputs[downsampled_idxs]

    def get_probabilities_and_forward_outputs(
        self, data: Union[torch.Tensor, dict], target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()
