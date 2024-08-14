from collections import deque

from modyn.config.schema.pipeline.trigger.drift.detection_window import (
    AmountWindowingStrategy,
)

from .window import DetectionWindows


class AmountDetectionWindows(DetectionWindows):
    def __init__(self, config: AmountWindowingStrategy):
        super().__init__()
        self.config = config

        # using maxlen the deque will automatically remove the oldest elements if the buffers are full
        self.current: deque[tuple[int, int]] = deque(maxlen=config.amount_cur)

        # If the reference window is bigger than the current window, we need a reservoir to store the
        # pushed out elements from the current window as we might still need them for the reference window. If a
        # trigger happens the whole reservoir and the current window will be copied/moved to the reference window.
        # Therefore the reservoir should be the size of the difference between the reference and current window.
        self.current_reservoir: deque[tuple[int, int]] = deque(maxlen=max(0, config.amount_ref - config.amount_cur))

        self.reference: deque[tuple[int, int]] = deque(maxlen=config.amount_ref)

        # In overlapping mode, we need a dedicated buffer to track new samples that are not yet in the reference buffer.
        # The current_ and current_reservoir_ are insufficient because, after a trigger, they will contain the same elements as before,
        # which prevents us from copying the current elements to the reference buffer without creating duplicates.
        # `exclusive_current` contains exactly the new elements that are not yet in the reference buffer.
        self.exclusive_current: deque[tuple[int, int]] = deque(
            maxlen=config.amount_ref if self.config.allow_overlap else 0
        )

    def inform_data(self, data: list[tuple[int, int]]) -> None:
        assert self.config.amount_cur

        if self.config.allow_overlap:
            # use the dedicated buffer that tracks the new elements to be copied to reference on trigger
            self.exclusive_current.extend(data)
        else:
            # use the existing buffers
            remaining_pushes = len(self.current) + len(data) - self.config.amount_cur

            # move data from current window to reservoir by first copying the oldest elements in the reservoir
            # and then later extending the current window with the new data automatically removing the oldest elements
            for pushed_out in self.current:
                if remaining_pushes == 0:
                    break
                self.current_reservoir.append(pushed_out)
                remaining_pushes -= 1

        self.current.extend(data)

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
