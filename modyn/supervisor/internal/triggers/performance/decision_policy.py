from abc import ABC, abstractmethod

from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicPerformanceThresholdCriterion,
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
        self.config = config

    def evaluate_decision(
        self,
        update_interval_samples: int,
        evaluation_scores: dict[str, float],
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        if mode == "hindsight":
            return evaluation_scores[self.config.metric] < self.config.metric_threshold

        return (evaluation_scores[self.config.metric] < self.config.metric_threshold) or (
            performance_tracker.forecast_next_performance(mode) < self.config.metric_threshold
        )


class DynamicPerformanceThresholdDecisionPolicy(PerformanceDecisionPolicy):
    """Decision policy that will make the binary trigger decisions based on a
    dynamic threshold.

    Value falls below the rolling average more than the allowed
    deviation.
    """

    def __init__(self, config: DynamicPerformanceThresholdCriterion):
        self.config = config

    def evaluate_decision(
        self,
        update_interval_samples: int,
        evaluation_scores: dict[str, float],
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        expected = performance_tracker.forecast_expected_performance(mode)
        deviation = expected - evaluation_scores[self.config.metric]

        if mode == "hindsight":
            if self.config.absolute:
                return deviation >= self.config.deviation
            return deviation >= self.config.deviation * expected

        if self.config.absolute:
            allowed_absolute_deviation = self.config.deviation
        else:
            allowed_absolute_deviation = self.config.deviation * expected

        return (deviation >= allowed_absolute_deviation) or (
            (expected - performance_tracker.forecast_next_performance(mode)) >= allowed_absolute_deviation
        )


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
            forecasted_misclassifications = self.cumulated_avoidable_misclassifications + round(
                forecast_new_avoidable_misclassifications
            )

            return forecasted_misclassifications >= self.config.avoidable_misclassification_threshold

        raise ValueError(f"Unknown mode: {mode}")

    def inform_trigger(self) -> None:
        """Resets the cumulated avoidable misclassifications."""
        self.cumulated_avoidable_misclassifications = 0
