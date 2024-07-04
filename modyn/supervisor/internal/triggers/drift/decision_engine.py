from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

from modyn.config.schema.pipeline.trigger.drift.metric import DynamicThresholdCriterion, ThresholdDecisionCriterion


class DriftDecisionEngine(ABC):
    """
    Decision policy that will make the binary is_drift decisions based on the similarity/distance metrics.

    Each drift decision wraps one DriftMetric and observes it's time series of distance values.
    """

    @abstractmethod
    def evaluate_decision(self, distance: float, raw_is_drift: bool) -> bool:
        """
        Evaluate the decision based on the distance value or the raw is_drift decision.

        Args:
            distance: The distance value of the metric.
            raw_is_drift: The raw is_drift decision of the metric.

        Returns:
            The final is_drift decision.
        """


class HypothesisTestDecisionEngine(DriftDecisionEngine):
    """
    Decision policy that will make the binary is_drift decisions based on the p-value of a hypothesis test.

    Each drift decision wraps one DriftMetric and observes it's time series of p-values.
    """

    def __init__(self, config: DynamicThresholdCriterion):
        self.config = config

    def evaluate_decision(self, distance: float, raw_is_drift: bool) -> bool:
        return raw_is_drift


class ThresholdDecisionEngine(DriftDecisionEngine):
    """
    Decision policy that will make the binary is_drift decisions based on a threshold.

    Each drift decision wraps one DriftMetric and observes it's time series of distance values.
    """

    def __init__(self, config: ThresholdDecisionCriterion):
        self.config = config

    def evaluate_decision(self, distance: float, raw_is_drift: bool) -> bool:
        return distance >= self.config.threshold


class DynamicDecisionEngine(DriftDecisionEngine):
    """
    Decision policy that will make the binary is_drift decisions based on a dynamic threshold.

    Each drift decision wraps one DriftMetric and observes it's time series of distance values.
    """

    def __init__(self, config: DynamicThresholdCriterion):
        self.config = config
        self.score_observations: Deque = deque(maxlen=self.config.window_size)

    def evaluate_decision(self, distance: float, raw_is_drift: bool) -> bool:
        num_more_extreme = sum(1 for score in self.score_observations if score >= distance)
        trigger = True
        if len(self.score_observations) > 0:
            perc = num_more_extreme / len(self.score_observations)
            trigger = perc < self.config.percentile_threshold

        self.score_observations.append(distance)
        return trigger
