import pathlib
from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass

from modyn.config.schema.pipeline import ModynPipelineConfig
from modyn.config.schema.system.config import ModynConfig
from modyn.supervisor.internal.triggers.models import TriggerPolicyEvaluationLog


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
    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        """The supervisor informs the Trigger about new data. In case the
        concrete Trigger implementation decides to trigger, we return a list of
        _indices into new_data_. This list contains the indices of all data
        points that cause a trigger. The list might be empty or only contain a
        single element, which concrete Triggers need to respect.

        Args:
            new_data: List of new data (keys, timestamps, labels)
            log: The log to store the trigger policy evaluation results to be able to verify trigger decisions

        Returns:
            triggering_indices: List of all indices that trigger training
        """

    # TODO: rename to "new_model"
    def inform_previous_model(self, previous_model_id: int) -> None:
        """The supervisor informs the Trigger about the model_id of the
        previous trigger."""
