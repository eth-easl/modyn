from abc import ABC, abstractmethod
from typing import Callable


class Trigger(ABC):
    def __init__(self, callback: Callable[[str, int], None], trigger_config: dict) -> None:
        assert callback is not None, "callback cannot be None."
        assert trigger_config is not None, "trigger_config cannot be None."
        self.callback = callback

    def inform(self, new_data: list[tuple[str, int]]) -> tuple[bool, int]:
        """The supervisor informs the trigger about new data.
        In case the concrete trigger decides to trigger, we call the callback
        as many times as the new data triggered training, with the data points that triggered the training.
        This blocks until training has finished, assuming that the callback blocks.

                Parameters:
                        new_data (list[tuple[str, int]]): List of new data. Can be empty.

                Returns:
                        triggered (bool): True if we triggered training, False otherwise.
        """

        triggers = self._decide_for_trigger(new_data)
        # We make sure to sort the triggers increasing by the timestamp, to avoid assumptions on the trigger implementation
        triggers.sort(key=lambda tup: tup[1])

        for key, timestamp in triggers:
            self.callback(key, timestamp)

        return len(triggers) > 0

    @abstractmethod
    def _decide_for_trigger(self, new_data: list[tuple[str, int]]) -> list[tuple[str, int]]:
        """Returns a list of all data points and their timestamps that cause a trigger
        We might trigger multiple times in case lots of new data came in since
        last inform.
        """
