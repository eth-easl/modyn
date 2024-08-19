# TODO
from __future__ import annotations

import logging
from collections.abc import Generator

from modyn.config.schema.pipeline.trigger.ensemble import EnsembleTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.data_amount import (
    DataAmountTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig
from modyn.supervisor.internal.triggers.amounttrigger import DataAmountTrigger
from modyn.supervisor.internal.triggers.timetrigger import TimeTrigger
from modyn.supervisor.internal.triggers.trigger import Trigger, TriggerContext
from modyn.supervisor.internal.triggers.utils.models import (
    TriggerPolicyEvaluationLog,
)

logger = logging.getLogger(__name__)


class EnsembleTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, config: EnsembleTriggerConfig):
        self.config = config
        self.subtriggers = self._create_subtriggers(config)

        # allows to evaluate triggers in a fixed interval
        self._sample_left_until_detection = config.detection_interval_data_points

        self._leftover_data: list[tuple[int, int]] = []
        """Stores data that was not processed in the last inform call because
        the detection interval was not filled."""

    def init_trigger(self, context: TriggerContext) -> None:
        for trigger in self.subtriggers.values():
            trigger.init_trigger(context)

    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        new_key_ts = self._leftover_data + [
            (key, timestamp) for key, timestamp, _ in new_data
        ]

        # reappending the leftover data to the new data requires incrementing the sample left until detection
        self._sample_left_until_detection += len(self._leftover_data)
        processing_head_in_batch = (
            0  # index of the first unprocessed data point in the batch
        )

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
            self._sample_left_until_detection = (
                self.config.detection_interval_data_points
            )

            # Delegate the detection to the subtriggers on our batch
            decisions = self._evaluate_subtriggers(next_detection_interval)
            aggregated_decision = self.config.ensemble_strategy.aggregate_decision_func(
                {trigger_id: decision[0] for trigger_id, decision in decisions.items()}
            )

            ensemble_eval_log = PerformanceTriggerEvalLog(
                triggered=triggered,
                trigger_index=trigger_idx,
                evaluation_interval=(
                    next_detection_interval[0][1],
                    next_detection_interval[-1][1],
                ),
                num_samples=num_samples,
                num_misclassifications=num_misclassifications,
                evaluation_scores=evaluation_scores,
                policy_decisions=policy_decisions,
            )
            if log:
                log.evaluations.append(ensemble_eval_log)

            self._last_detection_interval = next_detection_interval
            if triggered:
                yield trigger_idx

    def inform_previous_model(self, previous_model_id: int) -> None:
        for trigger in self.subtriggers.values():
            trigger.inform_previous_model(previous_model_id)

    # --------------------------------------------------- Internal --------------------------------------------------- #

    def _create_subtriggers(self, config: EnsembleTriggerConfig) -> dict[str, Trigger]:
        subtriggers: dict[str, Trigger] = {}
        for trigger_id, trigger_config in config.policies.items():
            if isinstance(trigger_config, TimeTriggerConfig):
                subtriggers[trigger_id] = TimeTrigger(trigger_config)
            elif isinstance(trigger_config, DataAmountTriggerConfig):
                subtriggers[trigger_id] = DataAmountTrigger(trigger_config)
            elif isinstance(trigger_config, DataDriftTrigger):
                subtriggers[trigger_id] = DataDriftTrigger(trigger_config)
            elif isinstance(trigger_config, EnsembleTriggerConfig):
                raise NotImplementedError("Nested ensemble triggers are not supported.")
            else:
                raise ValueError(f"Unknown trigger config: {trigger_config}")
        return subtriggers

    def _evaluate_subtriggers(
        self, new_data: list[tuple[int, int, int]]
    ) -> dict[str, tuple[bool, TriggerPolicyEvaluationLog]]:
        decisions = {}
        for trigger_id, trigger in self.subtriggers.items():
            log = TriggerPolicyEvaluationLog(trigger_id)
            decisions[trigger_id] = bool(list(trigger.inform(new_data, log)))
        return decisions
