from __future__ import annotations

import logging
from abc import abstractmethod

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.cost.cost import CostTriggerConfig
from modyn.supervisor.internal.triggers.batchedtrigger import BatchedTrigger
from modyn.supervisor.internal.triggers.cost.cost_tracker import CostTracker
from modyn.supervisor.internal.triggers.cost.incorporation_latency_tracker import (
    IncorporationLatencyTracker,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.models import (
    CostAwareTriggerEvalLog,
    TriggerPolicyEvaluationLog,
)

logger = logging.getLogger(__name__)


class CostTrigger(BatchedTrigger):
    """Triggers when a cumulated regret metric exceeds the estimated training
    time."""

    def __init__(self, config: CostTriggerConfig):
        super().__init__(config)
        self.config = config
        self.context: TriggerContext | None = None

        self._triggered_once = False
        self._previous_batch_end_time: int | None = None

        # cost information
        self._unincorporated_samples = 0
        self.cost_tracker = CostTracker(config.cost_tracking_window_size)
        self.latency_tracker = IncorporationLatencyTracker()
        """Maintains the regret metric and the cumulative regret latency,
        semantics are defined by the subclass."""

    @override
    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context

    @override
    def _evaluate_batch(
        self,
        batch: list[tuple[int, int]],
        trigger_candidate_idx: int,
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> bool:
        # Updates
        batch_start = self._previous_batch_end_time or batch[0][1]
        batch_duration = batch[-1][1] - batch_start
        self._previous_batch_end_time = batch[-1][1]
        self._unincorporated_samples += len(batch)

        traintime_estimate = -1.0

        regret_metric, regret_log = self._compute_regret_metric(batch, batch_start, batch_duration)
        regret_in_traintime_unit = regret_metric / self.config.conversion_factor

        # --------------------------------------------- Trigger Decision --------------------------------------------- #

        if (not self._triggered_once) or not self.warmup_trigger.completed:
            delegated_trigger_results = self.warmup_trigger.delegate_inform(batch)
            triggered = not self._triggered_once or delegated_trigger_results
            self._triggered_once = True

            # discard regret_metric

        else:
            traintime_estimate = self.cost_tracker.forecast_training_time(self._unincorporated_samples)
            triggered = regret_in_traintime_unit >= traintime_estimate

        # -------------------------------------------------- Log ------------------------------------------------- #

        drift_eval_log = CostAwareTriggerEvalLog(
            triggered=triggered,
            trigger_index=trigger_candidate_idx,
            num_samples=len(batch),
            evaluation_interval=(
                batch[0][1],
                batch[-1][1],
            ),
            regret_metric=regret_metric,
            regret_log=regret_log,
            traintime_estimate=traintime_estimate,
            regret_in_traintime_unit=regret_in_traintime_unit,
        )
        if log:
            log.evaluations.append(drift_eval_log)

        return triggered

    @override
    def inform_new_model(
        self,
        most_recent_model_id: int,
        number_samples: int | None = None,
        training_time: float | None = None,
    ) -> None:
        """Update the cost tracker with the new model metadata."""
        self._unincorporated_samples = 0
        assert not self._triggered_once or (
            number_samples is not None and training_time is not None
        ), "Only the pre-trained model is allowed to not supply the training time and number of samples"
        if number_samples and training_time:
            self.cost_tracker.inform_trigger(number_samples, training_time)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     INTERNAL                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    @abstractmethod
    def _compute_regret_metric(
        self, batch: list[tuple[int, int]], batch_start: int, batch_duration: int
    ) -> tuple[float, dict]:
        """Compute the regret metric for the current state of the trigger.

        This method will update the _incorporation_latency_tracker.

        Note: Child classes will use the arguments to build an internal state.

        Args:
            batch: The batch of data points to compute the regret metric for.
            batch_duration: The duration of the last period in seconds.

        Returns:
            regret_metric: The regret metric for the current state.
            additional_log: Additional log information to be stored in the evaluation log
        """
