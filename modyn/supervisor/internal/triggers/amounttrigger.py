from collections.abc import Generator

from modyn.config.schema.pipeline import DataAmountTriggerConfig
from modyn.supervisor.internal.triggers.trigger import Trigger
from modyn.supervisor.internal.triggers.utils.models import TriggerPolicyEvaluationLog


class DataAmountTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, config: DataAmountTriggerConfig):
        self.data_points_for_trigger: int = config.num_samples
        self.remaining_data_points = 0

        assert self.data_points_for_trigger > 0, "data_points_for_trigger needs to be at least 1"

        super().__init__()

    def inform(
        self, new_data: list[tuple[int, int, int]], log: TriggerPolicyEvaluationLog | None = None
    ) -> Generator[int, None, None]:
        assert self.remaining_data_points < self.data_points_for_trigger, "Inconsistent remaining datapoints"

        first_idx = self.data_points_for_trigger - self.remaining_data_points - 1
        triggering_indices = list(range(first_idx, len(new_data), self.data_points_for_trigger))

        self.remaining_data_points = (self.remaining_data_points + len(new_data)) % self.data_points_for_trigger

        yield from triggering_indices
