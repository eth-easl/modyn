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
    """Triggers when a certain number of data points have been used."""

    def __init__(self, config: AvoidableMisclassificationCostTriggerConfig):
        self.config = config
        self.context: TriggerContext | None = None

        self._sample_left_until_detection = config.detection_interval_data_points
        self._triggered_once = False

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     INTERNAL                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    @override
    def _compute_regret_metric(self, batch: list[tuple[int, int]], batch_duration: float) -> float:
        """Compute the regret metric for the current state of the trigger."""

        regret = len(batch)
        return self._incorporation_latency_tracker.add_latency(regret, batch_duration)
