from typing import Generator

from modyn.supervisor.internal.triggers.trigger import Trigger


class DataAmountTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, trigger_config: dict):
        if "data_points_for_trigger" not in trigger_config.keys():
            raise ValueError("Trigger config is missing `data_points_for_trigger` field")

        self.data_points_for_trigger: int = trigger_config["data_points_for_trigger"]
        assert self.data_points_for_trigger > 0, "data_points_for_trigger needs to be at least 1"
        self.remaining_data_points = 0

        super().__init__(trigger_config)

    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int, None, None]:
        assert self.remaining_data_points < self.data_points_for_trigger, "Inconsistent remaining datapoints"

        first_idx = self.data_points_for_trigger - self.remaining_data_points - 1
        triggering_indices = list(range(first_idx, len(new_data), self.data_points_for_trigger))

        self.remaining_data_points = (self.remaining_data_points + len(new_data)) % self.data_points_for_trigger

        yield from triggering_indices
