from abc import ABC, abstractmethod
from collections import deque


class DetectionWindows(ABC):
    """Wrapper and manager for the drift detection windows include reference,
    current and reservoir window.

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
        self.current: deque[tuple[int, int]] = deque()
        self.current_reservoir: deque[tuple[int, int]] = deque()
        self.reference: deque[tuple[int, int]] = deque()

    @abstractmethod
    def inform_data(self, data: list[tuple[int, int]]) -> None: ...

    @abstractmethod
    def inform_trigger(self) -> None: ...
