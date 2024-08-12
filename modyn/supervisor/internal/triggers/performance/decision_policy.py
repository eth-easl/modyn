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
    """
    Decision policy that will make the binary is_drift decisions on observations of a performance metric.
    """

    @abstractmethod
    def evaluate_decision(
        self,
        performance: float,
        mode: TriggerEvaluationMode,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
    ) -> bool:
        """
        Evaluate the decision based on the given observation.

        At the time of calling this the performance_tracker has already been updated with the new performance value
        to allow for forecast based decisions.

        Args:
            todo: todo

        Returns:
            The final is_drift decision.
        """


class StaticPerformanceThresholdDecisionPolicy(DriftDecisionPolicy):
    """
    Decision policy that will make the binary is_drift decisions based on a static threshold.
    """

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
            performance if mode == "hindsight" else performance_tracker.forecast_next_performance(mode)
        ) < self.config.metric_threshold


class DynamicPerformanceThresholdDecisionPolicy(DriftDecisionPolicy):
    """
    Decision policy that will make the binary is_drift decisions based on a dynamic threshold.
    """

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
            performance if mode == "hindsight" else performance_tracker.forecast_next_performance(mode)
        ) < performance_tracker.forecast_expected_performance(mode) - self.config.allowed_deviation


class StaticNumberAvoidableMisclassificationDecisionPolicy(DriftDecisionPolicy):
    """
    Decision policy that will make the binary is_drift decisions based on a static number of
    cumulated avoidable misclassifications.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold

    def evaluate_decision(
        self,
        performance: float,
        mode: TriggerEvaluationMode,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
    ) -> bool:
        # TODO: geogebra formula
        # TODO
        # TODO: next

        return (
            performance if mode == "hindsight" else performance_tracker.forecast_next_performance(mode)
        ) < performance_tracker.forecast_expected_performance(mode) - self.config.allowed_deviation
