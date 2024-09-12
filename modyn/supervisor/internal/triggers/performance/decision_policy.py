from abc import ABC, abstractmethod

from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicQuantilePerformanceThresholdCriterion,
    DynamicRollingAveragePerformanceThresholdCriterion,
    StaticNumberAvoidableMisclassificationCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.const.types import ForecastingMethod, TriggerEvaluationMode
from modyn.supervisor.internal.triggers.performance.data_density_tracker import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.misclassification_estimator import (
    NumberAvoidableMisclassificationEstimator,
)
from modyn.supervisor.internal.triggers.performance.performance_tracker import (
    PerformanceTracker,
)
from modyn.supervisor.internal.triggers.utils.decision_policy import (
    DynamicQuantileThresholdPolicy,
    DynamicRollingAverageThresholdPolicy,
    StaticThresholdDecisionPolicy,
)


class PerformanceDecisionPolicy(ABC):
    """Decision policy that will make the binary trigger decisions on
    observations of a performance metric."""

    @abstractmethod
    def evaluate_decision(
        self,
        update_interval_samples: int,
        evaluation_scores: dict[str, float],
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        """Evaluate the decision based on the given observation.

        At the time of calling this the performance_tracker has already been updated with the new performance value
        to allow for forecast based decisions.

        Also, data_density has to be updated with the new data interval.

        Args:
            update_interval_samples: The interval in which the decision is made.
            performance: The observed performance metric.
            mode: The mode in which the decision should be evaluated.
            data_density: The data density tracker, updated with the new data interval.
            performance_tracker: The performance tracker, updated with the new performance value.

        Returns:
            The final trigger decision.
        """

    def inform_trigger(self) -> None:
        """Inform the decision policy that a trigger has been invoked."""


class StaticPerformanceThresholdDecisionPolicy(PerformanceDecisionPolicy):
    """Decision policy that will make the binary trigger decisions based on a
    static threshold."""

    def __init__(self, config: StaticPerformanceThresholdCriterion):
        self.metric = config.metric
        self._wrapped = StaticThresholdDecisionPolicy(threshold=config.metric_threshold, triggering_direction="lower")

    def evaluate_decision(
        self,
        update_interval_samples: int,
        evaluation_scores: dict[str, float],
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        return self._wrapped.evaluate_decision(measurement=evaluation_scores[self.metric])


class DynamicPerformanceQuantileThresholdPolicy(PerformanceDecisionPolicy):
    """Wrapper for DynamicRollingAverageThresholdPolicy.

    Triggers if value is in the lower quantile of the rolling window.
    """

    def __init__(self, config: DynamicQuantilePerformanceThresholdCriterion):
        self.metric = config.metric
        self._wrapped = DynamicQuantileThresholdPolicy(
            window_size=config.window_size,
            quantile=config.quantile,
            triggering_direction="lower",
        )

    def evaluate_decision(
        self,
        update_interval_samples: int,
        evaluation_scores: dict[str, float],
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        return self._wrapped.evaluate_decision(measurement=evaluation_scores[self.metric])


class DynamicPerformanceRollingAverageThresholdPolicy(PerformanceDecisionPolicy):
    """Wrapper for DynamicRollingAverageThresholdPolicy.

    Trigger if value falls below the rolling average more than the
    allowed deviation.
    """

    def __init__(self, config: DynamicRollingAveragePerformanceThresholdCriterion):
        self.metric = config.metric
        self._wrapped = DynamicRollingAverageThresholdPolicy(
            window_size=config.window_size,
            deviation=config.deviation,
            absolute=config.absolute,
            triggering_direction="lower",
        )

    def evaluate_decision(
        self,
        update_interval_samples: int,
        evaluation_scores: dict[str, float],
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        return self._wrapped.evaluate_decision(measurement=evaluation_scores[self.metric])


class StaticNumberAvoidableMisclassificationDecisionPolicy(PerformanceDecisionPolicy):
    """Decision policy that will make the binary trigger decisions based on a
    static number of cumulated avoidable misclassifications."""

    def __init__(self, config: StaticNumberAvoidableMisclassificationCriterion):
        """
        Args:
            threshold: The threshold of cumulated avoidable misclassifications.
        """

        self.config = config
        self.cumulated_avoidable_misclassifications = 0
        self.misclassification_estimator = NumberAvoidableMisclassificationEstimator(
            expected_accuracy=config.expected_accuracy,
            allow_reduction=config.allow_reduction,
        )

    def evaluate_decision(
        self,
        update_interval_samples: int,
        evaluation_scores: dict[str, float],
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        """Utilizes the state of `DataDensityTracker` and `PerformanceTracker`
        to make the decision.

        We support both the "hindsight" and "forecast" mode.

        In the "hindsight" mode, the decision is made based on the current performance and the cumulated avoidable misclassifications.

        - Formalization:
            - historic observation:
                - data_cum_since_last_trigger: The cumulated data points since the last trigger.
                - avoidable_misclassifications_since_last_trigger: The cumulated avoidable misclassifications since
                    the last trigger.

        In the "lookahead" mode, the decision is made based on the current performance, the cumulated avoidable
        misclassifications, future performance estimates and future data density estimates.
        Similar to the "hindsight" mode, we first check if current performance already leads to a transgression
        of the threshold and therefore to a trigger.

        If that's not the case we estimate the cumulated avoidable misclassifications until the next point of update.
        If we expect a transgression of the threshold before the next update point, we trigger.

        This forward looking approach tries to avoid exceeding the misclassification budget in the first place.
        """

        new_avoidable_misclassifications, forecast_new_avoidable_misclassifications = (
            self.misclassification_estimator.estimate_avoidable_misclassifications(
                update_interval_samples=update_interval_samples,
                data_density=data_density,
                performance_tracker=performance_tracker,
                method=method,
            )
        )
        self.cumulated_avoidable_misclassifications += round(new_avoidable_misclassifications)
        hindsight_exceeded = (
            self.cumulated_avoidable_misclassifications >= self.config.avoidable_misclassification_threshold
        )

        if hindsight_exceeded:
            # past misclassifications already exceed the threshold, forecasting not needed
            return True

        if mode == "hindsight":
            return False

        if mode == "lookahead":
            # NOTE: this is a early feature that might not work as expected
            forecasted_misclassifications = self.cumulated_avoidable_misclassifications + round(
                forecast_new_avoidable_misclassifications
            )

            return forecasted_misclassifications >= self.config.avoidable_misclassification_threshold

        raise ValueError(f"Unknown mode: {mode}")

    def inform_trigger(self) -> None:
        """Resets the cumulated avoidable misclassifications."""
        self.cumulated_avoidable_misclassifications = 0
