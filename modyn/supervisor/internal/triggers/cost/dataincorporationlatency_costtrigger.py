from __future__ import annotations

import logging

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.cost.cost import (
    AvoidableMisclassificationCostTriggerConfig,
)
from modyn.supervisor.internal.triggers.cost.costtrigger import CostTrigger

logger = logging.getLogger(__name__)


class DataIncorporationLatencyCostTrigger(CostTrigger):
    """Triggers when the cumulated data incorporation latency (regret) exceeds
    the estimated training time."""

    def __init__(self, config: AvoidableMisclassificationCostTriggerConfig):
        super().__init__(config)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     INTERNAL                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    @override
    def _compute_regret_metric(
        self, batch: list[tuple[int, int]], batch_duration: float
    ) -> float:
        """Compute the regret metric for the current state of the trigger."""

        regret = len(batch)
        return self._incorporation_latency_tracker.add_latency(regret, batch_duration)
