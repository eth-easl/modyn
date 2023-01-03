from typing import Callable
from abc import ABC, abstractmethod


class Trigger(ABC):
    def __init__(self, callback: Callable, trigger_config: dict):
        assert callback is not None, "callback cannot be None."
        assert trigger_config is not None, "trigger_config cannot be None."
        self.callback = callback

    def inform(self, new_data: list[tuple[str, int]]) -> bool:
        """The supervisor regularly informs the trigger.
        This method ahould get called regularly by the supervisor, even if there was no new data.
        If there was any new data, then len(new_data) > 0.
        In case the concrete trigger decides to trigger, we call the callback as many times as the new data triggered training.
        This blocks until training has finished.

                Parameters:
                        new_data (list[tuple[str, int]]): List of new data. Can be empty.

                Returns:
                        triggered (bool): True if we triggered training, False otherwise.
        """

        num_triggers = self._decide_for_trigger(new_data)
        triggered = num_triggers > 0

        while num_triggers > 0:
            self.callback()
            num_triggers -= 1

        return triggered

    @abstractmethod
    def _decide_for_trigger(self, new_data: list[str, int]) -> int:
        """Returns how often we trigger, given the new data.
        We might trigger multiple times in case lots of new data came in since
        last inform.
        """
        pass
