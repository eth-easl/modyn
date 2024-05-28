from __future__ import annotations

import torch


class BatchAccumulator:
    """
    A class for accumulating batches of data over a specified period.

    Args:
        accumulation_period (int): The number of batches to accumulate before returning the accumulated batch.
        target_device (str): The target device to move the accumulated data to.

    """

    def __init__(self, accumulation_period: int, target_device: str):
        self._accumulation_buffer: list = []
        self._accumulation_period = accumulation_period
        self._current_batch_number = 0
        self._target_device = target_device

    def inform_batch(
        self,
        data: torch.Tensor | dict[str, torch.Tensor],
        sample_ids: list,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> bool:
        """
        Informs the accumulator about a new batch of data.

        Args:
            data (torch.Tensor | dict[str, torch.Tensor]): The input data for the batch.
            sample_ids (list): The sample IDs associated with the batch.
            target (torch.Tensor): The target values for the batch.
            weights (torch.Tensor): The weights for the batch.

        Returns:
            bool: True if the accumulation period is reached and the accumulated batch is ready, False otherwise.
        """
        data = {k: v.detach().cpu() for k, v in data.items()} if isinstance(data, dict) else data.detach().cpu()

        self._accumulation_buffer.append((data, sample_ids.copy(), target.detach().cpu(), weights.detach().cpu()))

        if (self._current_batch_number + 1) % self._accumulation_period == 0:
            self._current_batch_number += 1
            return True

        self._current_batch_number += 1
        return False

    def get_accumulated_batch(self) -> tuple[torch.Tensor | dict[str, torch.Tensor], list, torch.Tensor, torch.Tensor]:
        """
        Retrieves the accumulated batch of data.

        Returns:
            tuple[torch.Tensor | dict[str, torch.Tensor], list, torch.Tensor, torch.Tensor]:
                - The accumulated input data.
                - The accumulated sample IDs.
                - The accumulated target values.
                - The accumulated weights.
        """
        data, sample_ids, target, weights = self._accumulation_buffer[-1]

        for partial_data, partial_sids, partial_target, partial_weights in reversed(self._accumulation_buffer[:-1]):
            if isinstance(data, torch.Tensor):
                data = torch.cat((data, partial_data.to(self._target_device)))
            else:
                for key in data:
                    data[key] = torch.cat((data[key], partial_data[key].to(self._target_device)))

            sample_ids.extend(partial_sids)
            target = torch.cat((target, partial_target.to(self._target_device)))
            weights = torch.cat((weights, partial_weights.to(self._target_device)))

        self._accumulation_buffer.clear()

        return data, sample_ids, target, weights
