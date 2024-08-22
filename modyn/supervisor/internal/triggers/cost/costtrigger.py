from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Generator

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.cost.cost import CostTriggerConfig
from modyn.supervisor.internal.triggers.cost.cost_tracker import CostTracker
from modyn.supervisor.internal.triggers.cost.incorporation_latency_tracker import (
    IncorporationLatencyTracker,
)
from modyn.supervisor.internal.triggers.models import (
    CostAwareTriggerEvalLog,
    TriggerPolicyEvaluationLog,
)
from modyn.supervisor.internal.triggers.trigger import Trigger, TriggerContext

logger = logging.getLogger(__name__)


class CostTrigger(Trigger):
    """Triggers when a cumulated regret metric exceeds the estimated training
    time."""

    def __init__(self, config: CostTriggerConfig):
        self.config = config
        self.context: TriggerContext | None = None

        self._sample_left_until_detection = config.evaluation_interval_data_points
        self._triggered_once = False
        self._previous_batch_end_time: int | None = None
        self._leftover_data: list[tuple[int, int]] = []
        """Stores data that was not processed in the last inform call because
        the detection interval was not filled."""

        self._unincorporated_samples = 0
        self._cost_tracker = CostTracker(config.cost_tracking_window_size)
        self._incorporation_latency_tracker = IncorporationLatencyTracker()
        """Maintains the regret metric and the cumulative regret latency,
        semantics are defined by the subclass."""

    @override
    def init_trigger(self, context: TriggerContext) -> None:
        self.context = context

    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        new_key_ts = self._leftover_data + [(key, timestamp) for key, timestamp, _ in new_data]
        # reappending the leftover data to the new data requires incrementing the sample left until detection
        self._sample_left_until_detection += len(self._leftover_data)

        # index of the first unprocessed data point in the batch
        processing_head_in_batch = 0

        # Go through remaining data in new data in batches of `detect_interval`
        while True:
            if self._sample_left_until_detection - len(new_key_ts) > 0:
                # No detection in this trigger because of too few data points to fill detection interval
                self._leftover_data = new_key_ts
                self._sample_left_until_detection -= len(new_key_ts)
                return

            # At least one detection, fill up window up to that detection
            next_detection_interval = new_key_ts[: self._sample_left_until_detection]

            # Update the remaining data
            processing_head_in_batch += len(next_detection_interval)
            new_key_ts = new_key_ts[len(next_detection_interval) :]

            # Reset for next detection
            self._sample_left_until_detection = self.config.evaluation_interval_data_points

            # Updates
            batch_duration = next_detection_interval[-1][1] - (
                self._previous_batch_end_time or next_detection_interval[0][1]
            )
            self._previous_batch_end_time = next_detection_interval[-1][1]
            self._unincorporated_samples += len(next_detection_interval)

            # ----------------------------------------------- decision ----------------------------------------------- #
            regret_metric = self._compute_regret_metric(next_detection_interval, batch_duration)

            if not self._triggered_once:
                traintime_estimate = -1.0
                regret_in_traintime_unit = -1.0
                triggered = self._triggered_once = True
            else:
                traintime_estimate = self._cost_tracker.forecast_training_time(self._unincorporated_samples)

                regret_in_traintime_unit = regret_metric * self.config.conversion_factor

                triggered = regret_in_traintime_unit >= traintime_estimate

            # -------------------------------------------------- Log ------------------------------------------------- #

            trigger_idx = processing_head_in_batch - 1
            drift_eval_log = CostAwareTriggerEvalLog(
                triggered=triggered,
                trigger_index=trigger_idx,
                num_samples=len(next_detection_interval),
                evaluation_interval=(
                    next_detection_interval[0][1],
                    next_detection_interval[-1][1],
                ),
                regret_metric=regret_metric,
                traintime_estimate=traintime_estimate,
                regret_in_traintime_unit=regret_in_traintime_unit,
            )
            if log:
                log.evaluations.append(drift_eval_log)

            # ----------------------------------------------- Response ----------------------------------------------- #

            if triggered:
                yield trigger_idx

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
            self._cost_tracker.inform_trigger(number_samples, training_time)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     INTERNAL                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    @abstractmethod
    def _compute_regret_metric(self, batch: list[tuple[int, int]], batch_duration: float) -> float:
        """Compute the regret metric for the current state of the trigger.

        This method will update the _incorporation_latency_tracker.

        Args:
            batch: The batch of data points to compute the regret metric for.
            batch_duration: The duration of the last period in seconds.
        """
        ...
