from abc import ABC, abstractmethod

from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicPerformanceThresholdCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.const.types import TriggerEvaluationMode
from modyn.supervisor.internal.triggers.performance.data_density import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.performance import (
    PerformanceTracker,
)


class DriftDecisionPolicy(ABC):
    """Decision policy that will make the binary is_drift decisions on
    observations of a performance metric."""

    @abstractmethod
    def evaluate_decision(
        self,
        performance: float,
        mode: TriggerEvaluationMode,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
    ) -> bool:
        """Evaluate the decision based on the given observation.

        At the time of calling this the performance_tracker has already been updated with the new performance value
        to allow for forecast based decisions.

        Also, data_density has to be updated with the new data interval.

        Args:
            performance: The observed performance metric.
            mode: The mode in which the decision should be evaluated.
            data_density: The data density tracker, updated with the new data interval.
            performance_tracker: The performance tracker, updated with the new performance value.

        Returns:
            The final is_drift decision.
        """


class StaticPerformanceThresholdDecisionPolicy(DriftDecisionPolicy):
    """Decision policy that will make the binary is_drift decisions based on a
    static threshold."""

    def __init__(self, config: StaticPerformanceThresholdCriterion):
        self.config = config

    def evaluate_decision(
        self,
        performance: float,
        mode: TriggerEvaluationMode,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
    ) -> bool:
        return (
            performance
            if mode == "hindsight"
            else performance_tracker.forecast_next_performance(mode)
        ) < self.config.metric_threshold


class DynamicPerformanceThresholdDecisionPolicy(DriftDecisionPolicy):
    """Decision policy that will make the binary is_drift decisions based on a
    dynamic threshold."""

    def __init__(self, config: DynamicPerformanceThresholdCriterion):
        self.config = config

    def evaluate_decision(
        self,
        performance: float,
        mode: TriggerEvaluationMode,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
    ) -> bool:
        return (
            performance
            if mode == "hindsight"
            else performance_tracker.forecast_next_performance(mode)
        ) < performance_tracker.forecast_expected_performance(
            mode
        ) - self.config.allowed_deviation


class StaticNumberAvoidableMisclassificationDecisionPolicy(DriftDecisionPolicy):
    """Decision policy that will make the binary is_drift decisions based on a
    static number of cumulated avoidable misclassifications."""

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.cumulated_avoidable_misclassifications = 0

    def evaluate_decision(
        self,
        performance: float,
        mode: TriggerEvaluationMode,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
    ) -> bool:
        """Utilizes the state of `DataDensityTracker` and `PerformanceTracker` to make the decision.

        We support both the "hindsight" and "forecast" mode.

        In the "hindsight" mode, the decision is made based on the current performance and the cumulated avoidable misclassifications.

        - Formalization:
            - historic observation:
                - data_cum_since_last_trigger: The cumulated data points since the last trigger.
                - avoidable_misclassifications_since_last_trigger: The cumulated avoidable misclassifications since
                    the last trigger.

        In the "forecast" mode, the decision is made based on the current performance, the cumulated avoidable
        misclassifications, future performance estimates and future data density estimates.
        Similar to the "hindsight" mode, we first check if current performance already leads to a transgression
        of the threshold and therefore to a trigger.

        If that's not the case we estimate the cumulated avoidable misclassifications until the next point of update.
        If we expect a transgression of the threshold before the next update point, we trigger.

        This forward looking approach tries to avoid exceeding the misclassification budget in the first place.
        """
        last_interval_data_density = data_density.previous_batch_samples()
        last_interval_performance = performance_tracker.previous_performance()
        new_misclassifications = data_density.forecast_density(mode=mode)

        if mode == "hindsight":
            
        elif mode == "forecast":
            forcasted_data_density = data_density.forecast_density(mode=mode)
            forecasted_misclassifications = self.cumulated_avoidable_misclassifications + new_misclassifications
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        

        return (
            performance
            if mode == "hindsight"
            else performance_tracker.forecast_next_performance(mode)
        ) < performance_tracker.forecast_expected_performance(
            mode
        ) - self.config.allowed_deviation

    # TODO: inform trigger --> reset cumulative avoidable misclassifications
