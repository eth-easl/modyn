from typing import Callable

from modyn.backend.supervisor.internal.trigger import Trigger


class DataAmountTrigger(Trigger):
    """Triggers when a certain number of data points have been seen."""

    def __init__(self, callback: Callable, trigger_config: dict):
        assert (
            "data_points_for_trigger" in trigger_config.keys()
        ), "Trigger config is missing `data_points_for_trigger` field"

        self.data_points_for_trigger: int = trigger_config["data_points_for_trigger"]
        assert self.data_points_for_trigger > 0, "data_points_for_trigger needs to be at least 1"
        self.seen_data_points = 0

        super().__init__(callback, trigger_config)

    def _decide_for_trigger(self, new_data: list[tuple[str, int]]) -> list[int]:
        result: list[int] = []

        # TODO(MaxiBoether): efficiently implement triggering logic here.
        # Consider seen_data_points and then decide what data points in new_data cause a trigger
        # return the indices accordingly
        # Idea: cerate a numpy array where each entry represents its index (0, 1, 2,3,...) of length len(new_data)
        # Then add self.seen_data_points on each element (brotkasten)
        # Then do a modulo dpft on each element, get indices where True (= result)
        # self.seen_data_points = (self.seen_data_points + len(new_data)) mod self.data_points_for_trigger
        # fertig.

        return result
