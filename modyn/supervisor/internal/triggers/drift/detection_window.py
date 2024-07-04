from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

from modyn.config.schema.pipeline.trigger.drift.detection_window import AmountWindowingStrategy, TimeWindowingStrategy


class DetectionWindowManager(ABC):
    """
    Manager for the drift detection windows include reference, current and reservoir window.

    All windows contain tuples with (sample_id, timestamp).

    The manager is responsible for the following tasks:
    - Keep track of the current and reference window
    - Update the current window with new data
    - Move data from the current and reservoir window to the reference window

    If the reference window is bigger than the current window, we still want to fill up the whole reference window
    after a trigger by taking |reference| elements from the current window.
    We therefore need to keep track of the elements that exceed the current window but should still be transferred
    to the reference window. If something is is dropped from the reservoir, it won't ever be used again in any window.
    """

    def __init__(self) -> None:
        self.current_: Deque[tuple[int, int]] = deque()
        self.current_reservoir_: Deque[tuple[int, int]] = deque()
        self.reference_: Deque[tuple[int, int]] = deque()

    @abstractmethod
    def inform_data(self, data: list[tuple[int, int]]) -> None: ...

    @abstractmethod
    def inform_trigger(self) -> None: ...


class AmountDetectionWindowManager(DetectionWindowManager):
    def __init__(self, config: AmountWindowingStrategy):
        super().__init__()
        self.config = config
        assert config.current_buffer_size
        assert config.reference_buffer_size

        # using maxlen the deque will automatically remove the oldest elements if the buffers are full
        self.current_: Deque[tuple[int, int]] = deque(maxlen=config.current_buffer_size)
        self.current_reservoir_: Deque[tuple[int, int]] = deque(
            maxlen=config.reference_buffer_size - config.current_buffer_size
        )
        self.reference_: Deque[tuple[int, int]] = deque(maxlen=config.reference_buffer_size)

    def inform_data(self, data: list[tuple[int, int]]) -> None:
        assert self.config.current_buffer_size
        num_pushed_out = len(self.current_) + len(data) - self.config.current_buffer_size

        # move data from current window to reservoir
        for pushed_out in self.current_:
            if num_pushed_out == 0:
                break
            self.current_reservoir_.append(pushed_out)
            num_pushed_out -= 1

        self.current_.extend(data)

    def inform_trigger(self) -> None:
        # First, move data from the reference window to the reservoir
        self.reference_.extend(self.current_reservoir_)
        self.current_reservoir_.clear()

        # Move data from current to reference window
        self.reference_.extend(self.current_)

        if not self.config.allow_overlap:
            # if allow_overlap, don't reset the current window
            self.current_.clear()


class TimeDetectionWindowManager(DetectionWindowManager):
    def __init__(self, config: TimeWindowingStrategy):
        super().__init__()
        self.config = config
        self.current_: Deque[tuple[int, int]] = deque()
        self.current_reservoir_: Deque[tuple[int, int]] = deque()
        self.reference_: Deque[tuple[int, int]] = deque()

    def inform_data(self, data: list[tuple[int, int]]) -> None:
        if not data:
            return

        last_time = data[-1][1]

        # First, add the data to the current window, nothing will be pushed out automatically as there's no buffer limit
        self.current_.extend(data)

        # now, pop the data that is too old from the current window and move it to the reservoir.
        # This assumes that the data is sorted by timestamp.
        while self.current_ and self.current_[0][1] < last_time - self.config.limit_seconds_cur:
            self.current_reservoir_.append(self.current_.popleft())

        # next, we drop the data from the reservoir that is too old (forget them completely)
        while self.current_reservoir_ and self.current_reservoir_[0][1] < last_time - self.config.limit_seconds_ref:
            self.current_reservoir_.popleft()

    def inform_trigger(self) -> None:
        # First, move data from the reference window to the reservoir
        self.reference_.extend(self.current_reservoir_)
        self.current_reservoir_.clear()

        # Move data from current to reference window
        self.reference_.extend(self.current_)

        if not self.config.allow_overlap:
            # if allow_overlap, don't reset the current window
            self.current_.clear()

        # now, we drop the data from the reference window that is too old (forget them completely)
        if not self.reference_:
            return

        last_time = self.reference_[-1][1]
        while self.reference_ and self.reference_[0][1] < last_time - self.config.limit_seconds_ref:
            self.reference_.popleft()
