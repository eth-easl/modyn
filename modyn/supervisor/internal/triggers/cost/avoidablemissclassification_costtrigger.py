import logging

from typing_extensions import override

from modyn.config.schema.pipeline.trigger.cost.cost import (
    AvoidableMisclassificationCostTriggerConfig,
)
from modyn.supervisor.internal.triggers.cost.costtrigger import CostTrigger
from modyn.supervisor.internal.triggers.performance.misclassification_estimator import (
    NumberAvoidableMisclassificationEstimator,
)
from modyn.supervisor.internal.triggers.performance.performancetrigger_mixin import (
    PerformanceTriggerMixin,
)
from modyn.supervisor.internal.triggers.trigger import TriggerContext

logger = logging.getLogger(__name__)


class AvoidableMisclassificationCostTrigger(CostTrigger, PerformanceTriggerMixin):
    """Triggers when the avoidable misclassification cost incorporation latency
    (regret) exceeds the estimated training time."""

    def __init__(self, config: AvoidableMisclassificationCostTriggerConfig):
        CostTrigger.__init__(self, config)
        PerformanceTriggerMixin.__init__(self, config)

        self.config = config
        self.context: TriggerContext | None = None

        self.misclassification_estimator = NumberAvoidableMisclassificationEstimator(
            config.expected_accuracy, config.allow_reduction
        )

    @override
    def init_trigger(self, context: TriggerContext) -> None:
        # Call CostTrigger's init_trigger method to initialize the trigger context
        super().init_trigger(context)

        # Call PerformanceTriggerMixin's init_trigger method to initialize the internal performance detection state
        self._init_trigger(context)

    @override
    def inform_new_model(
        self,
        most_recent_model_id: int,
        number_samples: int | None = None,
        training_time: float | None = None,
    ) -> None:
        """Update the cost and performance trackers with the new model
        metadata."""

        # Call CostTrigger's inform_new_model method to update the cost tracker
        super().inform_new_model(most_recent_model_id, number_samples, training_time)

        # Call the internal PerformanceTriggerMixin's inform_new_model method to update the performance tracker
        self._inform_new_model(most_recent_model_id, self._last_detection_interval)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     INTERNAL                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    @override
    def _compute_regret_metric(self, batch: list[tuple[int, int]], batch_start: int, batch_duration: int) -> float:
        """Compute the regret metric for the current state of the trigger."""

        self.data_density.inform_data(batch)
        num_samples, num_misclassifications, evaluation_scores = self._run_evaluation(interval_data=batch)

        self.performance_tracker.inform_evaluation(
            num_samples=num_samples,
            num_misclassifications=num_misclassifications,
            evaluation_scores=evaluation_scores,
        )

        estimated_new_avoidable_misclassifications, _ = (
            self.misclassification_estimator.estimate_avoidable_misclassifications(
                update_interval_samples=self.config.evaluation_interval_data_points,
                data_density=self.data_density,
                performance_tracker=self.performance_tracker,
                method=self.config.forecasting_method,
            )
        )

        return self._incorporation_latency_tracker.add_latency(
            estimated_new_avoidable_misclassifications, batch_duration
        )
