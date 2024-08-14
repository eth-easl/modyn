from abc import ABC, abstractmethod

from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicPerformanceThresholdCriterion,
    StaticNumberAvoidableMisclassificationCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.const.types import ForecastingMethod, TriggerEvaluationMode
from modyn.supervisor.internal.triggers.performance.data_density import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.performance import (
    PerformanceTracker,
)


class PerformanceDecisionPolicy(ABC):
    """Decision policy that will make the binary trigger decisions on
    observations of a performance metric."""

    @abstractmethod
    def evaluate_decision(
        self,
        update_interval: int,
        performance: float,
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
            update_interval: The interval in which the decision is made.
            performance: The observed performance metric.
            mode: The mode in which the decision should be evaluated.
            data_density: The data density tracker, updated with the new data interval.
            performance_tracker: The performance tracker, updated with the new performance value.

        Returns:
            The final trigger decision.
        """

    def inform_trigger(self) -> None:
        """Inform the decision policy that a trigger has been invoked."""
        pass


class StaticPerformanceThresholdDecisionPolicy(PerformanceDecisionPolicy):
    """Decision policy that will make the binary trigger decisions based on a
    static threshold."""

    def __init__(self, config: StaticPerformanceThresholdCriterion):
        self.config = config

    def evaluate_decision(
        self,
        update_interval: int,
        performance: float,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        if mode == "hindsight":
            return performance < self.config.metric_threshold

        return (performance < self.config.metric_threshold) or (
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
        update_interval: int,
        performance: float,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        mode: TriggerEvaluationMode,
        method: ForecastingMethod,
    ) -> bool:
        threshold = performance_tracker.forecast_expected_performance(mode) - self.config.allowed_deviation

        if mode == "hindsight":
            return performance < threshold

        return (performance < threshold) or (performance_tracker.forecast_next_performance(mode) < threshold)


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

    # TODO: allow_reduction

    def evaluate_decision(
        self,
        update_interval: int,
        performance: float,
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

        # compute the number of avoidable misclassifications by retrieving the actual misclassifications
        # and the expected misclassifications through the expected accuracy for the last interval.
        previous_interval_num_misclassifications = performance_tracker.previous_batch_num_misclassifications()

        # the expected performance won't change unless there's a trigger
        expected_accuracy = (
            performance_tracker.forecast_expected_accuracy(method=method)
            if self.config.expected_accuracy is None
            else self.config.expected_accuracy
        )
        previous_interval_num_samples = data_density.previous_batch_samples()
        previous_expected_num_misclassifications = (1 - expected_accuracy) * previous_interval_num_samples
        new_avoidable_misclassifications = (
            previous_interval_num_misclassifications - previous_expected_num_misclassifications
        )
        if new_avoidable_misclassifications < 0 and not self.config.allow_reduction:
            new_avoidable_misclassifications = 0

        self.cumulated_avoidable_misclassifications += round(new_avoidable_misclassifications)

        if mode == "hindsight":
            return self.cumulated_avoidable_misclassifications >= self.config.avoidable_misclassification_threshold

        elif mode == "lookahead":
            # past misclassifications already exceed the threshold, forecasting not needed
            if self.cumulated_avoidable_misclassifications >= self.config.avoidable_misclassification_threshold:
                return True

            forecasted_data_density = data_density.forecast_density(method=method)
            forecast_accuracy = performance_tracker.forecast_next_accuracy(method=method)

            accuracy_delta = expected_accuracy - forecast_accuracy
            if accuracy_delta < 0 and not self.config.allow_reduction:
                accuracy_delta = 0

            # new misclassification = accuracy * samples; samples = data_density * interval_duration
            forecast_new_avoidable_misclassifications = accuracy_delta * forecasted_data_density * update_interval

            forecasted_misclassifications = (
                self.cumulated_avoidable_misclassifications + forecast_new_avoidable_misclassifications
            )

            return forecasted_misclassifications >= self.config.avoidable_misclassification_threshold

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def inform_trigger(self) -> None:
        """Resets the cumulated avoidable misclassifications."""
        self.cumulated_avoidable_misclassifications = 0
