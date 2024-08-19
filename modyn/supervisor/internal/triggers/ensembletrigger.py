# TODO
from __future__ import annotations

import logging
from collections.abc import Generator

from modyn.config.schema.pipeline.trigger.drift.config import DataDriftTriggerConfig
from modyn.config.schema.pipeline.trigger.ensemble import EnsembleTriggerConfig
from modyn.config.schema.pipeline.trigger.simple.data_amount import (
    DataAmountTriggerConfig,
)
from modyn.config.schema.pipeline.trigger.simple.time import TimeTriggerConfig
from modyn.supervisor.internal.triggers.amounttrigger import DataAmountTrigger
from modyn.supervisor.internal.triggers.datadrifttrigger import DataDriftTrigger
from modyn.supervisor.internal.triggers.timetrigger import TimeTrigger
from modyn.supervisor.internal.triggers.trigger import Trigger, TriggerContext
from modyn.supervisor.internal.triggers.utils.models import (
    EnsembleTriggerEvalLog,
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

        self._leftover_data: list[tuple[int, int, int]] = []
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
        new_key_ts = self._leftover_data + new_data

        # reappending the leftover data to the new data requires incrementing the sample left until detection
        self._sample_left_until_detection += len(self._leftover_data)
        processing_head_in_batch = 0  # index of the first unprocessed data point in the batch

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
            self._sample_left_until_detection = self.config.detection_interval_data_points

            # Delegate the detection to the subtriggers on our batch
            subtrigger_results = self._evaluate_subtriggers(next_detection_interval)

            triggered = self.config.ensemble_strategy.aggregate_decision_func(
                {trigger_id: decision[0] for trigger_id, decision in subtrigger_results.items()}
            )
            subtrigger_decisions = {trigger_id: decision[0] for trigger_id, decision in subtrigger_results.items()}
            subtrigger_indexes = {trigger_id: decision[1] for trigger_id, decision in subtrigger_results.items()}
            subtrigger_logs = {trigger_id: decision[2] for trigger_id, decision in subtrigger_results.items()}

            trigger_idx = processing_head_in_batch - 1
            ensemble_eval_log = EnsembleTriggerEvalLog(
                triggered=triggered,
                trigger_index=trigger_idx,
                evaluation_interval=(
                    next_detection_interval[0][1],
                    next_detection_interval[-1][1],
                ),
                subtrigger_decisions=subtrigger_decisions,
                subtrigger_indexes=subtrigger_indexes,
                subtrigger_logs=subtrigger_logs,
            )
            if log:
                log.evaluations.append(ensemble_eval_log)
            if triggered:
                yield trigger_idx

    def inform_previous_model(self, previous_model_id: int) -> None:
        for trigger in self.subtriggers.values():
            trigger.inform_previous_model(previous_model_id)

    # --------------------------------------------------- Internal --------------------------------------------------- #

    def _create_subtriggers(self, config: EnsembleTriggerConfig) -> dict[str, Trigger]:
        subtriggers: dict[str, Trigger] = {}
        for trigger_id, trigger_config in config.subtriggers.items():
            if isinstance(trigger_config, TimeTriggerConfig):
                subtriggers[trigger_id] = TimeTrigger(trigger_config)
            elif isinstance(trigger_config, DataAmountTriggerConfig):
                subtriggers[trigger_id] = DataAmountTrigger(trigger_config)
            elif isinstance(trigger_config, DataDriftTriggerConfig):
                subtriggers[trigger_id] = DataDriftTrigger(trigger_config)
            elif isinstance(trigger_config, EnsembleTriggerConfig):
                raise NotImplementedError("Nested ensemble triggers are not supported.")
            else:
                raise ValueError(f"Unknown trigger config: {trigger_config}")
        return subtriggers

    def _evaluate_subtriggers(
        self, new_data: list[tuple[int, int, int]]
    ) -> dict[str, tuple[bool, list[int], TriggerPolicyEvaluationLog]]:
        subtrigger_results: dict[str, tuple[bool, list[int], TriggerPolicyEvaluationLog]] = {}
        for trigger_id, trigger in self.subtriggers.items():
            log = TriggerPolicyEvaluationLog()
            sub_triggers = list(trigger.inform(new_data, log))
            sub_decision = bool(len(sub_triggers) > 0)
            subtrigger_results[trigger_id] = (sub_decision, sub_triggers, log)

        return subtrigger_results
