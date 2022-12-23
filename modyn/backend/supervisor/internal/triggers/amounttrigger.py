from modyn.backend.supervisor.internal.trigger import Trigger
from typing import Callable


class DataAmountTrigger(Trigger):
    """Triggers when a certain number of data point has been seen.
    """

    def __init__(self, callback: Callable, data_points_for_trigger: int):
        assert data_points_for_trigger > 0, "data_points_for_trigger needs to be at least 1"
        self.data_points_for_trigger = data_points_for_trigger
        self.seen_data_points = 0
        super().__init__(callback)

    def decide_for_trigger(self, new_data: list[tuple[str, int]]) -> int:
        self.seen_data_points += len(new_data)
        num_triggers = int(self.seen_data_points / self.data_points_for_trigger)
        self.seen_data_points -= self.data_points_for_trigger * num_triggers

        return num_triggers
