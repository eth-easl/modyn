from typing import Callable

from modyn.backend.supervisor.internal.trigger import Trigger

import numpy as np

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
        # TODO(Maxiboether): find numpy to do this directly, if possible
        triggering = np.array(range(self.remaining_data_points, len(new_data) + self.remaining_data_points)) % self.data_points_for_trigger
        triggering_indices = list(np.where(triggering)[0])

        self.remaining_data_points = (self.remaining_data_points + len(new_data)) % self.data_points_for_trigger

        return triggering_indices
