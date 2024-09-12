from __future__ import annotations

import logging

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicQuantilePerformanceThresholdCriterion,
    DynamicRollingAveragePerformanceThresholdCriterion,
    StaticNumberAvoidableMisclassificationCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.config.schema.pipeline.trigger.performance.performance import (
    PerformanceTriggerConfig,
)
from modyn.supervisor.internal.triggers.batchedtrigger import BatchedTrigger
from modyn.supervisor.internal.triggers.performance.decision_policy import (
    DynamicPerformanceQuantileThresholdPolicy,
    DynamicPerformanceRollingAverageThresholdPolicy,
    PerformanceDecisionPolicy,
    StaticNumberAvoidableMisclassificationDecisionPolicy,
    StaticPerformanceThresholdDecisionPolicy,
)
from modyn.supervisor.internal.triggers.performance.performancetrigger_mixin import (
    PerformanceTriggerMixin,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext
from modyn.supervisor.internal.triggers.utils.models import (
    PerformanceTriggerEvalLog,
    TriggerPolicyEvaluationLog,
)

logger = logging.getLogger(__name__)


class PerformanceTrigger(BatchedTrigger, PerformanceTriggerMixin):
    """Trigger based on the performance of the model.

    We support a simple performance drift approach that compares the
    most recent model performance with an expected performance value
    that can be static or dynamic through a rolling average.

    Additionally we support a regret based approach where the number of
    avoidable misclassifications (misclassifications that could have
    been avoided if we would have triggered) is compared to a threshold.
    """

    def __init__(self, config: PerformanceTriggerConfig) -> None:
        BatchedTrigger.__init__(self, config)
        PerformanceTriggerMixin.__init__(self, config)

        self.config = config
        self.context: TriggerContext | None = None

        self.decision_policies = _setup_decision_policies(config)

        self._triggered_once = False

    @override
    def init_trigger(self, context: TriggerContext) -> None:
        # Call PerformanceTriggerMixin's init_trigger method to initialize the internal performance detection state
        PerformanceTriggerMixin._init_trigger(self, context)

    @override
    def _evaluate_batch(
        self,
        batch: list[tuple[int, int]],
        trigger_candidate_idx: int,
        log: TriggerPolicyEvaluationLog | None = None,
    ) -> bool:
        # Run the evaluation (even if we don't use the result, e.g. for the first forced trigger)
        self.data_density.inform_data(batch)
        policy_decisions: dict[str, bool] = {}

        model_id = None
        num_samples = None
        num_misclassifications = None
        evaluation_scores: dict[str, float] = {}

        if self._triggered_once:
            # we can run a (warmup) evaluation, iff we have already triggered once and therefore a model at hand
            model_id, num_samples, num_misclassifications, evaluation_scores = PerformanceTriggerMixin._run_evaluation(
                self, interval_data=batch
            )

            # Let's update the state; in the warmup phase this will calibrate the policy without the decision
            # policies being consulted
            self.performance_tracker.inform_evaluation(
                num_samples=num_samples,
                num_misclassifications=num_misclassifications,
                evaluation_scores=evaluation_scores,
            )

        # --------------------------------------------- Trigger Decision --------------------------------------------- #

        if (not self._triggered_once) or not self.warmup_trigger.completed:
            # delegate to the warmup policy
            delegated_trigger_results = self.warmup_trigger.delegate_inform(batch)
            triggered = not self._triggered_once or delegated_trigger_results
            self._triggered_once = True

            # we have already reported the warmup evaluation results

        else:
            # evaluate the decision policies
            triggered = False
            for policy_name, policy in self.decision_policies.items():
                policy_decisions[policy_name] = policy.evaluate_decision(
                    update_interval_samples=self.config.evaluation_interval_data_points,
                    evaluation_scores=evaluation_scores,
                    data_density=self.data_density,
                    performance_tracker=self.performance_tracker,
                    mode=self.config.mode,
                    method=self.config.forecasting_method,
                )
                if policy_decisions[policy_name]:
                    triggered |= True

        if triggered:
            for policy in self.decision_policies.values():
                policy.inform_trigger()  # resets the internal state (e.g. misclassification counters)

        # -------------------------------------------------- Log ------------------------------------------------- #

        drift_eval_log = PerformanceTriggerEvalLog(
            triggered=triggered,
            trigger_index=trigger_candidate_idx,
            evaluation_interval=(batch[0][1], batch[-1][1]),
            id_model=model_id or -1,
            num_samples=num_samples or -1,
            num_misclassifications=num_misclassifications or -1,
            evaluation_scores=evaluation_scores or {},
            policy_decisions=policy_decisions,
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
        # Delegate to internal implementation
        PerformanceTriggerMixin._inform_new_model(self, most_recent_model_id, self._last_detection_interval)


def _setup_decision_policies(
    config: PerformanceTriggerConfig,
) -> dict[str, PerformanceDecisionPolicy]:
    """Policy factory that creates the decision policies based on the given
    configuration."""
    policies: dict[str, PerformanceDecisionPolicy] = {}
    for name, criterion in config.decision_criteria.items():
        if isinstance(criterion, StaticPerformanceThresholdCriterion):
            policies[name] = StaticPerformanceThresholdDecisionPolicy(criterion)
        elif isinstance(criterion, DynamicQuantilePerformanceThresholdCriterion):
            policies[name] = DynamicPerformanceQuantileThresholdPolicy(criterion)
        elif isinstance(criterion, DynamicRollingAveragePerformanceThresholdCriterion):
            policies[name] = DynamicPerformanceRollingAverageThresholdPolicy(criterion)
        elif isinstance(criterion, StaticNumberAvoidableMisclassificationCriterion):
            policies[name] = StaticNumberAvoidableMisclassificationDecisionPolicy(criterion)
    return policies
