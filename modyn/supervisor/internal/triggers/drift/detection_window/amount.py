from collections import deque
from typing import Deque

from modyn.config.schema.pipeline.trigger.drift.detection_window import AmountWindowingStrategy

from .window import DetectionWindowManager


class AmountDetectionWindowManager(DetectionWindowManager):
    def __init__(self, config: AmountWindowingStrategy):
        super().__init__()
        self.config = config
        assert config.current_buffer_size
        assert config.reference_buffer_size

        # using maxlen the deque will automatically remove the oldest elements if the buffers are full
        self.current_: Deque[tuple[int, int]] = deque(maxlen=config.current_buffer_size)
        self.current_reservoir_: Deque[tuple[int, int]] = deque(
            maxlen=max(0, config.reference_buffer_size - config.current_buffer_size)
        )
        self.reference_: Deque[tuple[int, int]] = deque(maxlen=config.reference_buffer_size)

        # in overlapping mode (we need dedicated buffer to keep track of the new samples that are not in
        # the reference buffer, yet). The current_ and current_reservoir_ are not enough as after
        # a trigger they will contain the same elements as before hindering us from copying the
        # current elements to the reference buffer (creating duplicates)
        self.exclusive_current: Deque[tuple[int, int]] = deque(
            maxlen=config.reference_buffer_size if self.config.allow_overlap else 0
        )

    def inform_data(self, data: list[tuple[int, int]]) -> None:
        assert self.config.current_buffer_size

        if self.config.allow_overlap:
            # use the dedicated buffer that tracks the new elements to be copied to reference on trigger
            self.exclusive_current.extend(data)
        else:
            # use the existing buffers
            num_pushed_out = len(self.current_) + len(data) - self.config.current_buffer_size

            # move data from current window to reservoir
            for pushed_out in self.current_:
                if num_pushed_out == 0:
                    break
                self.current_reservoir_.append(pushed_out)
                num_pushed_out -= 1

        self.current_.extend(data)

    def inform_trigger(self) -> None:
        if self.config.allow_overlap:
            # move all new elements to the reference buffer
            self.reference_.extend(self.exclusive_current)
            self.exclusive_current.clear()

        else:
            # First, move data from the reservoir window to the reference window
            self.reference_.extend(self.current_reservoir_)
            self.current_reservoir_.clear()

            # Move data from current to reference window
            self.reference_.extend(self.current_)

            # if allow_overlap, don't reset the current window
            self.current_.clear()
