from abc import ABC, abstractmethod
from collections import deque
from typing import Literal

import numpy as np


class DecisionPolicy(ABC):
    """Decision policy that will make the binary triggering decisions based on
    the similarity/measurement metrics.

    Note:
        In the case of data drift each decision policy wraps one DriftMetric and observes its time
        series of distance values.
    """

    def __init__(self, triggering_direction: Literal["higher", "lower"]):
        """
        Args:
            config: The configuration of the decision policy.
            triggering_direction: Whether a higher score should produce a trigger or a lower score.
        """
        self.triggering_direction = triggering_direction

    @abstractmethod
    def evaluate_decision(self, measurement: float) -> bool:
        """Evaluate the decision based on the measurement value or the raw
        triggering decision.

        Args:
            measurement: The measurement value of the metric.

        Returns:
            The final triggering decision.
        """


class StaticThresholdDecisionPolicy(DecisionPolicy):
    """Decision policy that will make the binary triggering decisions based on
    a threshold."""

    def __init__(self, threshold: float, triggering_direction: Literal["higher", "lower"]):
        """
        Args:
            threshold: The threshold that results in a trigger.
            triggering_direction: Whether a higher score should produce a trigger or a lower score.
        """
        super().__init__(triggering_direction)
        self.threshold = threshold

    def evaluate_decision(self, measurement: float) -> bool:
        return measurement >= self.threshold if self.triggering_direction == "higher" else measurement <= self.threshold


class DynamicDecisionPolicy(DecisionPolicy):
    """Decision policy that will make the binary triggering decisions based on
    a dynamic threshold."""

    def __init__(self, window_size: int, triggering_direction: Literal["higher", "lower"]):
        """Reusable decision policy that will make the binary triggering
        decisions based on a dynamic threshold.

        Args:
            window_size: The size of the observations to be considered for the decision.
            triggering_direction: Whether a higher score should produce a trigger or a lower score.
        """
        super().__init__(triggering_direction)
        self.score_observations: deque = deque(maxlen=window_size)


class DynamicQuantileThresholdPolicy(DynamicDecisionPolicy):
    """Dynamic threshold based on a extremeness quantile of the previous
    measurement values.

    We compare a new measurement value with the series of previous measurement values
    and decide if it's more extreme than a certain quantile of the series. Therefore we count the
    `num_more_extreme` values that are greater than the new measurement and compare it with the
    `quantile` threshold.
    """

    def __init__(
        self,
        window_size: int,
        quantile: float,
        triggering_direction: Literal["higher", "lower"],
    ):
        """
        Args:
            window_size: The size of the observations to be considered for the decision.
            quantile: The quantile that a threshold has to be in to trigger event.
            triggering_direction: Whether a higher score should produce a trigger or a lower score.
        """
        super().__init__(window_size, triggering_direction)
        self.quantile = quantile

    def evaluate_decision(self, measurement: float) -> bool:
        if len(self.score_observations) == 0:
            self.score_observations.append(measurement)
            return True

        # let's linearly interpolate to find the most extreme self.quantile value
        quantile_threshold = (
            # direction: higher --> higher is worse and invokes a trigger
            # most extreme 5% --> numpy quantile 0.95
            1 - self.quantile
            if self.triggering_direction == "higher"
            # direction: lower --> lower is worse and invokes a trigger
            # most extreme 5% --> numpy quantile 0.05
            else self.quantile
        )
        threshold = float(np.quantile(a=self.score_observations, q=quantile_threshold, method="linear"))

        self.score_observations.append(measurement)

        return measurement > threshold if self.triggering_direction == "higher" else measurement < threshold


class DynamicRollingAverageThresholdPolicy(DynamicDecisionPolicy):
    """Triggers when a new measurement value deviates from the rolling average
    by a certain amount or percentage."""

    def __init__(
        self,
        window_size: int,
        deviation: float,
        absolute: bool,
        triggering_direction: Literal["higher", "lower"],
    ):
        """
        Args:
            window_size: The size of the observations to be considered for the decision.
            deviation: The deviation from the rolling average that triggers.
            absolute: Whether the deviation is absolute or relative to the rolling average.
            triggering_direction: Whether a higher score should produce a trigger or a lower score.
        """
        super().__init__(window_size, triggering_direction)
        self.window_size = window_size
        self.deviation = deviation
        self.absolute = absolute

    def evaluate_decision(self, measurement: float) -> bool:
        if not self.score_observations:
            self.score_observations.append(measurement)
            return True

        rolling_average = sum(self.score_observations) / len(self.score_observations)
        deviation = (
            measurement - rolling_average if self.triggering_direction == "higher" else rolling_average - measurement
        )

        self.score_observations.append(measurement)

        if self.absolute:
            return deviation >= self.deviation
        return deviation >= self.deviation * rolling_average
