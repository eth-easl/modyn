import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

from modyn.config.schema.config import ModynConfig
from modyn.config.schema.pipeline import ModynPipelineConfig


@dataclass
class TriggerContext:
    pipeline_id: int
    pipeline_config: ModynPipelineConfig
    modyn_config: ModynConfig
    base_dir: pathlib.Path


class Trigger(ABC):

    # pylint: disable=unnecessary-pass
    def init_trigger(self, context: TriggerContext) -> None:
        """The supervisor initializes the concrete Trigger with Trigger-type-specific configurations
        base_dir: the base directory to store Trigger outputs. A location at the supervisor.
        """
        pass

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

    # pylint: disable=unnecessary-pass
    def inform_previous_trigger_and_data_points(self, previous_trigger_id: int, data_points: int) -> None:
        """The supervisor informs the Trigger about the previous trigger_id
        and data points in the previous trigger."""
        pass

    # pylint: disable=unnecessary-pass
    def inform_previous_model(self, previous_model_id: int) -> None:
        """The supervisor informs the Trigger about the model_id of the previous trigger"""
        pass
