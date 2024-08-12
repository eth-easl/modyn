from abc import ABC, abstractmethod
from collections import deque

from modyn.config.schema.pipeline.trigger.drift.metric import (
    DynamicThresholdCriterion,
    ThresholdDecisionCriterion,
)


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
    dynamic threshold.

    We compare a new distance value with the series of previous distance values
    and decide if it's more extreme than a certain percentile of the series. Therefore we count the
    `num_more_extreme` values that are greater than the new distance and compare it with the
    `percentile` threshold.

    TODO: we might want to also support some rolling average policy that will trigger if a distance is deviates
    from the average by a certain amount.
    """

    def __init__(self, config: DynamicThresholdCriterion):
        self.config = config
        self.score_observations: deque = deque(maxlen=self.config.window_size)

    def evaluate_decision(self, distance: float) -> bool:
        num_more_extreme = sum(1 for score in self.score_observations if score >= distance)
        trigger = True
        if len(self.score_observations) > 0:
            perc = num_more_extreme / len(self.score_observations)
            trigger = perc < self.config.percentile

        self.score_observations.append(distance)
        return trigger
