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
        self.leftover_data: list[tuple[str, int]] = []

        super().__init__(callback, trigger_config)

    def _decide_for_trigger(self, new_data: list[tuple[str, int]]) -> list[tuple[str, int]]:
        new_data.sort(key=lambda tup: tup[1])
        self.leftover_data.extend(new_data)

        result: list[tuple[str, int]] = []

        for i in range(0, len(self.leftover_data), self.data_points_for_trigger):
            sublist = self.leftover_data[i : i + self.data_points_for_trigger]

            if len(sublist) == self.data_points_for_trigger:
                # We got a trigger
                result.append(sublist[-1])
            else:
                # Last iteration, update leftover data
                assert i == len(self.leftover_data) - 1, "Data slicing failed"
                self.leftover_data = sublist

        return result
