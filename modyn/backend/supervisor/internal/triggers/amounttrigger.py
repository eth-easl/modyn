from typing import Callable

import numpy as np
from modyn.backend.supervisor.internal.trigger import Trigger


class DataAmountTrigger(Trigger):
    """Triggers when a certain number of data points have been seen."""

    def __init__(self, trigger_config: dict):
        assert (
            "data_points_for_trigger" in trigger_config.keys()
        ), "Trigger config is missing `data_points_for_trigger` field"

        self.data_points_for_trigger: int = trigger_config["data_points_for_trigger"]
        assert self.data_points_for_trigger > 0, "data_points_for_trigger needs to be at least 1"
        self.remaining_data_points = 0

        super().__init__(trigger_config)

    def inform(self, new_data: list[tuple[str, int]]) -> list[int]:
        # Imagine that we want to trigger every third data point
        # The idea is to get the indices as follows:
        # 0 1 2 3 4 5 6 7 8 9 10 => data point 2, 5, and 8 should trigger
        # We add 1 onto that index array: 1 2 3 4 5 6 7 8 9 10 11
        # We take that mod 3: 1 2 0 1 2 0 1 2 0 1 2. We see that all indices that are 0 mod 3 are the indices we are searching for.
        # This also works with remaining data, just add 1 + remaining data.

        triggering = (
            np.arange(1 + self.remaining_data_points, len(new_data) + self.remaining_data_points + 1)
            % self.data_points_for_trigger
        ) == 0
        triggering_indices = np.ravel(np.argwhere(triggering))

        self.remaining_data_points = (self.remaining_data_points + len(new_data)) % self.data_points_for_trigger

        return list(triggering_indices)
