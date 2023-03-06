from abc import ABC, abstractmethod


class Trigger(ABC):
    def __init__(self, trigger_config: dict) -> None:
        assert trigger_config is not None, "trigger_config cannot be None."

    @abstractmethod
    def inform(self, new_data: list[tuple[int, int, int]]) -> list[int]:
        """The supervisor informs the trigger about new data.
        In case the concrete trigger implementation decides to trigger, we return a list of _indices into new_data_.
        This list contains the indices of all data points that cause a trigger.
        The list might be empty or only contain a single element, which concrete triggers need to respect.

             Parameters:
                     new_data (list[tuple[str, int, int]]): List of new data (keys, timestamps, labels). Can be empty.

             Returns:
                     triggering_indices (list[int]): List of all indices that trigger training
        """
