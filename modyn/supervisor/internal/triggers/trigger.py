import pathlib
from typing import Generator
from abc import ABC, abstractmethod


class Trigger(ABC):
    def __init__(self, trigger_config: dict) -> None:
        assert trigger_config is not None, "trigger_config cannot be None."
    
    @abstractmethod
    def init_trigger(
        self, pipeline_id: int, pipeline_config: dict, modyn_config: dict, base_dir: pathlib.Path
    ) -> None:
        """The supervisor initializes the trigger with trigger type specific configurations"""

    @abstractmethod
    def inform(self, new_data: list[tuple[int, int, int]]) -> Generator[int, None, None]:
        """The supervisor informs the Trigger about new data.
        In case the concrete Trigger implementation decides to trigger, we return a list of _indices into new_data_.
        This list contains the indices of all data points that cause a trigger.
        The list might be empty or only contain a single element, which concrete Triggers need to respect.

             Parameters:
                     new_data (list[tuple[str, int, int]]): List of new data (keys, timestamps, labels). Can be empty.

             Returns:
                     triggering_indices (list[int]): List of all indices that trigger training
        """
    
    @abstractmethod
    def inform_previous_trigger(self, previous_trigger_id: int) -> None:
        """The supervisor informs the Trigger about the previous trigger_id"""

    @abstractmethod
    def inform_previous_model(self, previous_model_id: int) -> None:
        """The supervisor informs the Trigger about the model_id of the previous trigger"""
