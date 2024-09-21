from __future__ import annotations

import logging

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.cost.cost import (
    AvoidableMisclassificationCostTriggerConfig,
)
from modyn.supervisor.internal.triggers.costtrigger import CostTrigger

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
        self, batch: list[tuple[int, int]], batch_start: int, batch_duration: int
    ) -> tuple[float, dict]:
        """Compute the regret metric for the current state of the trigger."""

        new_regret = self.latency_tracker.add_latencies(
            regrets=[(time, 1.0) for _, time in batch],
            start_timestamp=batch_start,
            batch_duration=batch_duration,
        )
        regret_log = {
            "regret": new_regret,
            "batch_duration": batch_duration,
            "latency_tracker": new_regret,
        }
        return new_regret, regret_log
