from typing import Callable
from abc import ABC, abstractmethod


class Trigger(ABC):
    def __init__(self, callback: Callable):
        assert callback is not None, "callback cannot be None."
        self.callback = callback

    def inform(self, new_data: list[tuple[str, int]]) -> None:
        """The supervisor regularly informs the trigger.
        This method gets called regularly, even if there was no new data.
        If there was any new data, then len(new_data) > 0.
        new_data is a list of data keys and timestamps.
        In case the concrete trigger decides to trigger, we
        call the callback as many times as the new
        data triggered training.
        """

        num_triggers = self.decide_for_trigger(new_data)

        while num_triggers > 0:
            self.callback()
            num_triggers -= 1
            
            
    @abstractmethod
    def decide_for_trigger(self, new_data: list[str, int]) -> int:
        """Returns how often we trigger, given the new data.
        We might trigger multiple times in case lots of new data came in since
        last inform.
        """
        pass
