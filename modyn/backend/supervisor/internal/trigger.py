from abc import ABC, abstractmethod


class Trigger(ABC):
    def __init__(self, trigger_config: dict) -> None:
        assert trigger_config is not None, "trigger_config cannot be None."

    @abstractmethod
    def inform(self, new_data: list[tuple[str, int]]) -> list[int]:
        """The supervisor informs the trigger about new data.
        In case the concrete trigger implementation decides to trigger, we return a list
        of _indices into new_data_. This list contains the indices of all data points that
        cause a trigger. This list can be of length 0 or 1.

             Parameters:
                     new_data (list[tuple[str, int]]): List of new data. Can be empty.

             Returns:
                     triggering_indices (list[int]): List of all indices that trigger training
        """
