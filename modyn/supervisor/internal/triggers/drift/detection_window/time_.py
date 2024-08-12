from collections import deque

from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    TimeWindowingStrategy,
)

from .window import DetectionWindows


class TimeDetectionWindows(DetectionWindows):
    def __init__(self, config: TimeWindowingStrategy):
        super().__init__()
        self.config = config

        # in overlapping mode (we need dedicated buffer to keep track of the new samples that are not in
        # the reference buffer, yet). The current_ and current_reservoir_ are not enough as after
        # a trigger they will contain the same elements as before hindering us from copying the
        # current elements to the reference buffer (creating duplicates)
        self.exclusive_current: deque[tuple[int, int]] = deque()

    def inform_data(self, data: list[tuple[int, int]]) -> None:
        if not data:
            return

        last_time = data[-1][1]

        # First, add the data to the current window, nothing will be pushed out automatically as there's no buffer limit
        self.current.extend(data)

        if self.config.allow_overlap:
            # now, pop the data that is too old from the current window.
            # This assumes that the data is sorted by timestamp.
            while self.current and self.current[0][1] < last_time - self.config.limit_seconds_cur:
                self.current.popleft()

            self.exclusive_current.extend(data)

            # pop the data that is not in the reference scope anymore
            while self.exclusive_current and self.exclusive_current[0][1] < last_time - self.config.limit_seconds_ref:
                self.exclusive_current.popleft()

        else:
            # now, pop the data that is too old from the current window and move it to the reservoir.
            # This assumes that the data is sorted by timestamp.
            while self.current and self.current[0][1] < last_time - self.config.limit_seconds_cur:
                self.current_reservoir.append(self.current.popleft())

            # next, we drop the data from the reservoir that is too old (forget them completely)
            while self.current_reservoir and self.current_reservoir[0][1] < last_time - self.config.limit_seconds_ref:
                self.current_reservoir.popleft()

    def inform_trigger(self) -> None:
        if self.config.allow_overlap:
            # move all new elements to the reference buffer
            self.reference.extend(self.exclusive_current)
            self.exclusive_current.clear()

        else:
            # First, move data from the reservoir window to the reference window
            self.reference.extend(self.current_reservoir)
            self.current_reservoir.clear()

            # Move data from current to reference window
            self.reference.extend(self.current)
            self.current.clear()

        # now, we drop the data from the reference window that is too old (forget them completely)
        if not self.reference:
            return

        last_time = self.reference[-1][1]
        while self.reference and self.reference[0][1] < last_time - self.config.limit_seconds_ref:
            self.reference.popleft()
