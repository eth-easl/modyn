from abc import ABC, abstractmethod
from collections import deque

from modyn.config.schema.pipeline.trigger.drift.criterion import DynamicThresholdCriterion, ThresholdDecisionCriterion


class DriftDecisionPolicy(ABC):
    """Decision policy that will make the binary is_drift decisions based on
    the similarity/distance metrics.

    Each drift decision wraps one DriftMetric and observes its time
    series of distance values.
    """

    @abstractmethod
    def evaluate_decision(self, distance: float) -> bool:
        """Evaluate the decision based on the distance value or the raw
        is_drift decision.

        Args:
            distance: The distance value of the metric.

        Returns:
            The final is_drift decision.
        """


class ThresholdDecisionPolicy(DriftDecisionPolicy):
    """Decision policy that will make the binary is_drift decisions based on a
    threshold."""

    def __init__(self, config: ThresholdDecisionCriterion):
        self.config = config

    def evaluate_decision(self, distance: float) -> bool:
        return distance >= self.config.threshold


class DynamicDecisionPolicy(DriftDecisionPolicy):
    """Decision policy that will make the binary is_drift decisions based on a
    dynamic threshold."""

    def __init__(self, config: DynamicThresholdCriterion):
        self.config = config
        self.score_observations: deque = deque(maxlen=self.config.window_size)


class DynamicPercentileThresholdPolicy(DynamicDecisionPolicy):
    """Dynamic threshold based on a extremeness percentile of the previous
    distance values.

    We compare a new distance value with the series of previous distance values
    and decide if it's more extreme than a certain percentile of the series. Therefore we count the
    `num_more_extreme` values that are greater than the new distance and compare it with the
    `percentile` threshold.
    """

    def evaluate_decision(self, distance: float) -> bool:
        if len(self.score_observations) == 0:
            self.score_observations.append(distance)
            return True

        sorted_observations = list(sorted(self.score_observations))

        threshold = sorted_observations[
            min(
                max(
                    0,
                    int(round(len(sorted_observations) * (1.0 - self.config.percentile)))
                    - 1,  # from length to index space
                ),
                len(sorted_observations) - 1,
            )
        ]
        self.score_observations.append(distance)

        return distance > threshold


class DynamicRollingAverageThresholdPolicy(DynamicDecisionPolicy):
    """Triggers when a new distance value deviates from the rolling average by
    a certain amount or percentage."""

    def evaluate_decision(self, distance: float) -> bool:
        if not self.score_observations:
            self.score_observations.append(distance)
            return True

        rolling_average = sum(self.score_observations) / len(self.score_observations)
        deviation = distance - rolling_average

        self.score_observations.append(distance)

        if self.config.absolute:
            return deviation >= self.config.deviation
        return deviation >= self.config.deviation * rolling_average
