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


class DataDriftTrigger(Trigger):
    """Triggers when a certain number of data points have been used."""

    def __init__(self, config: EnsembleTriggerConfig):
        self.config = config
        self.subtriggers = self._create_subtriggers(config)

    def init_trigger(self, context: TriggerContext) -> None:
        for trigger in self.subtriggers.values():
            trigger.init_trigger(context)

    def inform(
        self,
        new_data: list[tuple[int, int, int]],
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> Generator[int, None, None]:
        decisions = self._evaluate_subtriggers(new_data)
        aggregated_decision = self.config.ensemble_strategy.aggregate_decision_func(
            {trigger_id: decision[0] for trigger_id, decision in decisions.items()}
        )
        
        
        

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
