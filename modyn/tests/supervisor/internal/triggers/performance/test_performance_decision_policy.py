from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from modyn.config.schema.pipeline.trigger.performance.criterion import (
    DynamicRollingAveragePerformanceThresholdCriterion,
    StaticNumberAvoidableMisclassificationCriterion,
    StaticPerformanceThresholdCriterion,
)
from modyn.supervisor.internal.triggers.performance.data_density_tracker import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.decision_policy import (
    DynamicPerformanceRollingAverageThresholdPolicy,
    StaticNumberAvoidableMisclassificationDecisionPolicy,
    StaticPerformanceThresholdDecisionPolicy,
)
from modyn.supervisor.internal.triggers.performance.performance_tracker import (
    PerformanceTracker,
)


@pytest.fixture
def static_threshold_policy() -> StaticPerformanceThresholdDecisionPolicy:
    return StaticPerformanceThresholdDecisionPolicy(
        config=StaticPerformanceThresholdCriterion(metric_threshold=0.8, metric="acc")
    )


@pytest.fixture
def dynamic_threshold_policy() -> DynamicPerformanceRollingAverageThresholdPolicy:
    return DynamicPerformanceRollingAverageThresholdPolicy(
        config=DynamicRollingAveragePerformanceThresholdCriterion(
            window_size=10, deviation=0.2, metric="acc", absolute=True
        )
    )


@pytest.fixture
def misclassification_criterion() -> StaticNumberAvoidableMisclassificationCriterion:
    # Using the config model criterion instead of the final policy to allow for adjustments of the config
    # in the tests before instantiating the policy
    return StaticNumberAvoidableMisclassificationCriterion(
        allow_reduction=True, avoidable_misclassification_threshold=10
    )


# -------------------------------------------------------------------------------------------------------------------- #
#                                                        Dummies                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


@pytest.fixture
def dummy_data_density_tracker() -> DataDensityTracker:
    return DataDensityTracker(window_size=3)


@pytest.fixture
def dummy_performance_tracker() -> PerformanceTracker:
    return PerformanceTracker(trigger_eval_window_size=3)


# -------------------------------------------------------------------------------------------------------------------- #
#                                       StaticPerformanceThresholdDecisionPolicy                                       #
# -------------------------------------------------------------------------------------------------------------------- #


def test_static_performance_hindsight(
    dummy_data_density_tracker: DataDensityTracker,
    dummy_performance_tracker: PerformanceTracker,
    static_threshold_policy: StaticPerformanceThresholdDecisionPolicy,
) -> None:
    """Test static threshold decision policy in hindsight mode."""
    eval_decision_kwargs = {
        "update_interval_samples": 10,
        "data_density": dummy_data_density_tracker,
        "performance_tracker": dummy_performance_tracker,
        "mode": "hindsight",
        "method": "rolling_average",
    }
    assert static_threshold_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0.79})
    assert not static_threshold_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0.81})

    with pytest.raises(KeyError):
        static_threshold_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"NOT_ACC": 0.81})


# -------------------------------------------------------------------------------------------------------------------- #
#                                       DynamicPerformanceThresholdDecisionPolicy                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def test_dynamic_performance_hindsight(
    dummy_performance_tracker: PerformanceTracker,
    dummy_data_density_tracker: DataDensityTracker,
    dynamic_threshold_policy: DynamicPerformanceRollingAverageThresholdPolicy,
) -> None:
    """Test dynamic threshold decision policy in hindsight mode."""

    eval_decision_kwargs = {
        "update_interval_samples": 10,
        "data_density": dummy_data_density_tracker,
        "performance_tracker": dummy_performance_tracker,
        "mode": "hindsight",
        "method": "rolling_average",
    }
    # current performance already below threshold, trigger independent of forecast
    assert dynamic_threshold_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0.5 - 0.21})
    assert not dynamic_threshold_policy.evaluate_decision(
        **eval_decision_kwargs, evaluation_scores={"acc": 0.5 - 0.19}
    )  # allowed deviation not reached

    with pytest.raises(KeyError):
        dynamic_threshold_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"NOT_ACC": 0.5})


# -------------------------------------------------------------------------------------------------------------------- #
#                                 StaticNumberAvoidableMisclassificationDecisionPolicy                                 #
# -------------------------------------------------------------------------------------------------------------------- #


@patch.object(
    DataDensityTracker,
    "previous_batch_num_samples",
    side_effect=[100, 100, 100, 100, 100],
    new_callable=PropertyMock,
)
@patch.object(
    PerformanceTracker,
    "forecast_expected_accuracy",
    side_effect=[1.0, 0.95, 1.0, 1.0, 1.0],
)
@patch.object(PerformanceTracker, "forecast_next_accuracy", return_value=-100)  # unused dummy
@patch.object(
    PerformanceTracker,
    "previous_batch_num_misclassifications",
    side_effect=[5, 9, 1, 9, 4],
    new_callable=PropertyMock,
)
def test_misclassification_hindsight(
    mock_previous_batch_num_misclassifications: MagicMock,
    mock_forecast_next_accuracy: MagicMock,
    mock_forecast_expected_accuracy: MagicMock,
    mock_previous_batch_num_samples: MagicMock,
    dummy_performance_tracker: PerformanceTracker,
    dummy_data_density_tracker: DataDensityTracker,
    misclassification_criterion: StaticNumberAvoidableMisclassificationCriterion,
) -> None:
    """Test static number avoidable misclassification policy in hindsight
    mode."""
    misclassification_policy = StaticNumberAvoidableMisclassificationDecisionPolicy(config=misclassification_criterion)

    eval_decision_kwargs = {
        "update_interval_samples": 10,
        "data_density": dummy_data_density_tracker,
        "performance_tracker": dummy_performance_tracker,
        "mode": "hindsight",
        "method": "rolling_average",
    }

    assert misclassification_policy.cumulated_avoidable_misclassifications == 0

    # don't trigger below the 10th misclassification

    # observed misclassifications=5, with expected accuracy of 1.0 every misclassification is avoidable
    assert not misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 5
    assert mock_previous_batch_num_misclassifications.call_count == 1

    # observed misclassifications=8, expected misclassifications: 5 --> 9-5=4 avoidable misclassifications
    # cumulated_avoidable_misclassifications: 5+4
    assert not misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 5 + 4
    assert mock_previous_batch_num_misclassifications.call_count == 2

    # observed misclassifications: 1, with expected accuracy of 1.0 every misclassification is avoidable
    assert misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 10
    assert mock_previous_batch_num_misclassifications.call_count == 3

    misclassification_policy.inform_trigger()  # reset misclassifications
    assert misclassification_policy.cumulated_avoidable_misclassifications == 0

    # observed misclassifications: 9, with expected accuracy of 1.0 every misclassification is avoidable
    assert not misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 9
    assert mock_previous_batch_num_misclassifications.call_count == 4

    # observed misclassifications: 4, with expected accuracy of 1.0 every misclassification is avoidable
    assert misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 13
    assert mock_previous_batch_num_misclassifications.call_count == 5


@patch.object(
    DataDensityTracker,
    "previous_batch_num_samples",
    side_effect=[100, 100, 100],
    new_callable=PropertyMock,
)
@patch.object(PerformanceTracker, "forecast_expected_accuracy", side_effect=[0, 0, 0])
@patch.object(PerformanceTracker, "forecast_next_accuracy", return_value=-100)  # unused dummy
@patch.object(
    PerformanceTracker,
    "previous_batch_num_misclassifications",
    side_effect=[90, 70, 70],
    new_callable=PropertyMock,
)
def test_misclassification_static_expected_performance(
    mock_previous_batch_num_misclassifications: MagicMock,
    mock_forecast_next_accuracy: MagicMock,
    mock_forecast_expected_accuracy: MagicMock,
    mock_previous_batch_num_samples: MagicMock,
    dummy_performance_tracker: PerformanceTracker,
    dummy_data_density_tracker: DataDensityTracker,
    misclassification_criterion: StaticNumberAvoidableMisclassificationCriterion,
) -> None:
    """Test static number avoidable misclassification policy in hindsight
    mode."""
    misclassification_criterion.expected_accuracy = 0.25
    misclassification_criterion.allow_reduction = True
    misclassification_policy = StaticNumberAvoidableMisclassificationDecisionPolicy(config=misclassification_criterion)

    eval_decision_kwargs = {
        "update_interval_samples": 10,
        "data_density": dummy_data_density_tracker,
        "performance_tracker": dummy_performance_tracker,
        "mode": "hindsight",
        "method": "rolling_average",
    }
    # expected misclassifications = (1-0.25)*100 = 75, observed misclassifications = 90
    assert misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 15
    assert mock_previous_batch_num_misclassifications.call_count == 1
    assert mock_previous_batch_num_samples.call_count == 1

    misclassification_policy.inform_trigger()  # reset misclassifications
    misclassification_policy.cumulated_avoidable_misclassifications = 10

    # negative avoidable misclassifications
    # expected misclassifications = (1-0.25)*100 = 75, observed misclassifications = 70
    assert not misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 5
    assert mock_previous_batch_num_misclassifications.call_count == 2
    assert mock_previous_batch_num_samples.call_count == 2

    # forbid reduction: with reduction cumulated_avoidable_misclassifications reduced from 5 to 0, here constant at 5
    misclassification_criterion.allow_reduction = False
    misclassification_policy = StaticNumberAvoidableMisclassificationDecisionPolicy(config=misclassification_criterion)
    misclassification_policy.cumulated_avoidable_misclassifications = 5
    assert not misclassification_policy.evaluate_decision(**eval_decision_kwargs, evaluation_scores={"acc": 0})
    assert misclassification_policy.cumulated_avoidable_misclassifications == 5
    assert mock_previous_batch_num_misclassifications.call_count == 3
    assert mock_previous_batch_num_samples.call_count == 3

    # not used in the case of static expected performance
    assert mock_forecast_expected_accuracy.call_count == 0
