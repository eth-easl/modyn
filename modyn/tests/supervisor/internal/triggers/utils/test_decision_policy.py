import pytest

from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicPercentileThresholdCriterion,
)
from modyn.supervisor.internal.triggers.utils.decision_policy import (
    DynamicPercentileThresholdPolicy,
    DynamicRollingAverageThresholdPolicy,
    StaticThresholdDecisionPolicy,
)


def test_threshold_decision_policy() -> None:
    policy = StaticThresholdDecisionPolicy(threshold=0.5, triggering_direction="higher")

    assert policy.evaluate_decision(0.6)
    assert not policy.evaluate_decision(0.4)


@pytest.mark.parametrize("percentile", [0.1, 0.5, 0.9])
def test_dynamic_decision_policy_initial(percentile: float) -> None:
    policy = DynamicPercentileThresholdPolicy(window_size=3, percentile=percentile, triggering_direction="higher")

    # Initially, the deque is empty, so any value should trigger
    assert policy.evaluate_decision(0.5)


def test_dynamic_decision_policy_with_observations() -> None:
    policy = DynamicPercentileThresholdPolicy(window_size=4, percentile=0.5, triggering_direction="higher")

    # Add initial observations
    policy.score_observations.extend([0.4, 0.5, 0.6, 0.7])

    # Testing with various distances
    assert not policy.evaluate_decision(0.3)  # Less than all observations
    assert policy.evaluate_decision(0.8)  # Greater than all observations
    assert not policy.evaluate_decision(0.6)


def test_dynamic_decision_policy_window_size() -> None:
    policy = DynamicPercentileThresholdPolicy(window_size=3, percentile=0.5, triggering_direction="higher")

    # Add observations to fill the window
    policy.evaluate_decision(0.4)
    policy.evaluate_decision(0.6)
    policy.evaluate_decision(0.7)

    # Adding another observation should remove the oldest one (0.4)
    assert policy.evaluate_decision(0.8)  # Greater than all observations
    assert len(policy.score_observations) == 3  # Ensure the deque is still at max length


def test_dynamic_decision_policy_percentile() -> None:
    config = DynamicPercentileThresholdCriterion(window_size=4, percentile=0.25)
    policy = DynamicPercentileThresholdPolicy(
        window_size=config.window_size,
        percentile=config.percentile,
        triggering_direction="higher",
    )

    # Add observations
    policy.evaluate_decision(0.4)
    policy.evaluate_decision(0.6)
    policy.evaluate_decision(0.7)
    policy.evaluate_decision(0.9)

    assert not policy.evaluate_decision(0.5)
    assert policy.evaluate_decision(0.8)
    assert not policy.evaluate_decision(0.7)


def test_dynamic_decision_policy_average() -> None:
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
