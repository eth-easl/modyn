import random

import torch


def _shuffle_list_and_tensor(samples: list, weights: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    num_elements = len(samples)
    indices = list(range(num_elements))
    random.shuffle(indices)

    shuffled_samples = [samples[i] for i in indices]
    shuffled_weights = weights[indices]

    return shuffled_samples, shuffled_weights
