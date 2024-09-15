import pytest

from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicQuantileThresholdCriterion,
)
from modyn.supervisor.internal.triggers.utils.decision_policy import (
    DynamicQuantileThresholdPolicy,
    DynamicRollingAverageThresholdPolicy,
    StaticThresholdDecisionPolicy,
)


def test_threshold_decision_policy() -> None:
    policy = StaticThresholdDecisionPolicy(threshold=0.5, triggering_direction="higher")

    assert policy.evaluate_decision(0.6)
    assert not policy.evaluate_decision(0.4)


@pytest.mark.parametrize("quantile", [0.1, 0.5, 0.9])
def test_dynamic_decision_policy_initial(quantile: float) -> None:
    policy = DynamicQuantileThresholdPolicy(window_size=3, quantile=quantile, triggering_direction="higher")

    # Initially, the deque is empty, so any value should trigger
    assert policy.evaluate_decision(0.5)


def test_dynamic_decision_policy_with_observations() -> None:
    policy = DynamicQuantileThresholdPolicy(window_size=4, quantile=0.5, triggering_direction="higher")

    # Add initial observations
    policy.score_observations.extend([0.4, 0.5, 0.6, 0.7])

    # Testing with various distances
    assert not policy.evaluate_decision(0.3)  # Less than all observations
    assert policy.evaluate_decision(0.8)  # Greater than all observations
    assert not policy.evaluate_decision(0.6)


def test_dynamic_decision_policy_window_size() -> None:
    policy = DynamicQuantileThresholdPolicy(window_size=3, quantile=0.5, triggering_direction="higher")

    # Add observations to fill the window
    policy.evaluate_decision(0.4)
    policy.evaluate_decision(0.6)
    policy.evaluate_decision(0.7)

    # Adding another observation should remove the oldest one (0.4)
    assert policy.evaluate_decision(0.8)  # Greater than all observations
    assert len(policy.score_observations) == 3  # Ensure the deque is still at max length


def test_dynamic_decision_policy_quantile_trigger_on_high_value() -> None:
    config = DynamicQuantileThresholdCriterion(window_size=4, quantile=0.25)
    policy = DynamicQuantileThresholdPolicy(
        window_size=config.window_size,
        quantile=config.quantile,
        triggering_direction="higher",
    )

    # Add observations (metric: e.g. distance, higher is worse)
    policy.evaluate_decision(5)
    policy.evaluate_decision(11)
    policy.evaluate_decision(15)
    policy.evaluate_decision(10)

    # observing 4 values with linear interpolation: 5 -> 0%, 10 -> 1/3, 11 -> 2/3, 15 -> 100%
    assert not policy.evaluate_decision(10)  # most extreme 25% threshold: >11, <15
    assert not policy.evaluate_decision(11.1)  # most extreme 25% threshold: >11, <15
    assert policy.evaluate_decision(14.5)  # most extreme 25% threshold: >11.1, <15


def test_dynamic_decision_policy_quantile_trigger_on_low_value() -> None:
    config = DynamicQuantileThresholdCriterion(window_size=3, quantile=0.25)
    policy = DynamicQuantileThresholdPolicy(
        window_size=config.window_size,
        quantile=config.quantile,
        triggering_direction="lower",
    )

    # Add observations (metric: e.g. accuracy, lower is worse)
    policy.evaluate_decision(0.9)
    policy.evaluate_decision(0.7)
    policy.evaluate_decision(0.5)

    # observing 3 values with linear interpolation: 0.5 -> 0%, 0.7 -> 50%, 0.9 -> 100%
    assert not policy.evaluate_decision(0.9)
    assert not policy.evaluate_decision(0.61)
    # 0.5 -> 0%, 0.61 -> 50%, 0.9 -> 100%: 25% is therefore in the middle of 0.5 and 0.61
    assert policy.evaluate_decision(0.53)
    assert not policy.evaluate_decision(0.59)


def test_dynamic_decision_policy_average_absolute() -> None:
    policy = DynamicRollingAverageThresholdPolicy(
        window_size=2, deviation=0.1, absolute=True, triggering_direction="higher"
    )

    # Add observations
    policy.evaluate_decision(1.0)
    policy.evaluate_decision(0.6)
    policy.evaluate_decision(0.7)
    policy.evaluate_decision(0.9)

    assert not policy.evaluate_decision(0.7)  # avg: 0.8
    assert not policy.evaluate_decision(0.8)  # avg: 0.8 (not >=0.1 deviation)
    assert not policy.evaluate_decision(0.85)  # avg: 0.75


def test_dynamic_decision_policy_average_relative() -> None:
    policy = DynamicRollingAverageThresholdPolicy(
        window_size=2, deviation=0.21, absolute=False, triggering_direction="lower"
    )

    # Add observations
    policy.evaluate_decision(1.0)
    policy.evaluate_decision(0.6)
    policy.evaluate_decision(0.3)
    policy.evaluate_decision(0.7)

    assert not policy.evaluate_decision(0.4)  # avg: 0.5 --> threshold: 0.5 * (1-0.21) = 0.395
    assert policy.evaluate_decision(0.43)  # avg: (0.4+0.7)/2 = 0.55 --> threshold: 0.55 * (1-0.21) = 0.4345
