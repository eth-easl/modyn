from __future__ import annotations

import logging

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.cost.cost import (
    AvoidableMisclassificationCostTriggerConfig,
)
from modyn.supervisor.internal.triggers.cost.costtrigger import CostTrigger
from modyn.supervisor.internal.triggers.trigger import TriggerContext

logger = logging.getLogger(__name__)


class AvoidableMisclassificationCostTrigger(CostTrigger):
    """Triggers when the avoidable misclassification cost incorporation latency
    (regret) exceeds the estimated training time."""

    def __init__(self, config: AvoidableMisclassificationCostTriggerConfig):
        self.config = config
        self.context: TriggerContext | None = None

        self._sample_left_until_detection = config.evaluation_interval_data_points
        self._triggered_once = False

    @override
    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context

    @override
    def inform_new_model(
        self,
        most_recent_model_id: int,
        number_samples: int | None = None,
        training_time: float | None = None,
    ) -> None:
        """Update the cost and performance trackers with the new model
        metadata."""
        super().inform_new_model(most_recent_model_id, number_samples, training_time)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     INTERNAL                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    @override
    def _compute_regret_metric(self, batch: list[tuple[int, int]], batch_duration: float) -> float:
        """Compute the regret metric for the current state of the trigger."""
        raise NotImplementedError()
