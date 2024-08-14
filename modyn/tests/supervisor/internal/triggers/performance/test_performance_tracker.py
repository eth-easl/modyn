import pytest

from modyn.supervisor.internal.triggers.performance.performance_tracker import (
    PerformanceTracker,
)


def test_initial_state() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    with pytest.raises(IndexError):
        tracker.previous_batch_num_misclassifications()
    assert len(tracker.trigger_evaluation_memory) == 0
    assert len(tracker.since_last_trigger) == 0


def test_inform_evaluation() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_evaluation(10, 2, {"acc": 0.8})
    assert tracker.previous_batch_num_misclassifications() == 2
    assert len(tracker.since_last_trigger) == 1
    assert tracker.since_last_trigger[0] == (10, 2, {"acc": 0.8})


def test_inform_trigger() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_evaluation(10, 2, {"acc": 0.8})
    tracker.inform_trigger(10, 1, {"acc": 0.9})
    assert tracker.previous_batch_num_misclassifications() == 1
    assert len(tracker.trigger_evaluation_memory) == 1
    assert tracker.trigger_evaluation_memory[-1] == (10, 1, {"acc": 0.9})
    assert len(tracker.since_last_trigger) == 1  # Reset after trigger


def test_inform_trigger_memory_rollover() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_trigger(10, 2, {"acc": 0.8})
    tracker.inform_trigger(20, 3, {"acc": 0.85})
    tracker.inform_trigger(10, 1, {"acc": 0.9})
    tracker.inform_trigger(20, 1, {"acc": 0.95})  # This should push out the first evaluation
    assert tracker.previous_batch_num_misclassifications() == 1
    assert len(tracker.trigger_evaluation_memory) == 3
    # First entry should be the second trigger
    assert tracker.trigger_evaluation_memory[0] == (20, 3, {"acc": 0.85})


def test_forecast_expected_performance_no_trigger() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    with pytest.raises(AssertionError):
        tracker.forecast_expected_performance(metric="acc")


def test_forecast_expected_performance_simple_average() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_trigger(10, 2, {"acc": 0.8})
    tracker.inform_trigger(20, 3, {"acc": 0.85})
    performance = tracker.forecast_expected_performance(metric="acc", method="rolling_average")
    assert performance == 0.825  # Simple average of 0.8 and 0.85
    performance = tracker.forecast_expected_performance(metric="acc", method="ridge_regression")
    # Simple average of 0.8 and 0.85 (not enough samples for ridge regression)
    assert performance == 0.825

    with pytest.raises(KeyError):
        tracker.forecast_expected_performance(metric="NOT_ACC", method="ridge_regression")


def test_forecast_expected_performance_ridge_regression() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=5)
    tracker.inform_trigger(10, 2, {"acc": 0.8})
    tracker.inform_trigger(20, 3, {"acc": 0.85})
    tracker.inform_trigger(10, 1, {"acc": 0.9})
    tracker.inform_trigger(20, 1, {"acc": 0.95})
    tracker.inform_trigger(10, 0, {"acc": 1.0})
    tracker.inform_trigger(20, 0, {"acc": 1.0})
    performance = tracker.forecast_expected_performance(metric="acc", method="rolling_average")
    assert isinstance(performance, float)  # Should run ridge regression
    assert performance == (0.85 + 0.9 + 0.95 + 1.0 + 1.0) / 5
    performance_ridge = tracker.forecast_expected_performance(metric="acc", method="ridge_regression")
    assert 0.9 <= performance_ridge


def test_forecast_next_performance_no_trigger() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    with pytest.raises(AssertionError):
        tracker.forecast_next_performance(metric="acc", method="ridge_regression")


def test_forecast_next_performance_simple_average() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_evaluation(10, 2, {"acc": 0.8})
    tracker.inform_evaluation(20, 3, {"acc": 0.85})
    performance = tracker.forecast_next_performance(metric="acc", method="rolling_average")
    assert performance == 0.825  # Simple average of 0.8 and 0.85


def test_forecast_next_performance_ridge_regression() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=5)
    tracker.inform_evaluation(10, 1, {"acc": 0.8})
    tracker.inform_evaluation(10, 1, {"acc": 0.85})
    tracker.inform_evaluation(10, 1, {"acc": 0.9})
    tracker.inform_evaluation(10, 1, {"acc": 0.95})
    tracker.inform_evaluation(10, 1, {"acc": 1.0})
    performance = tracker.forecast_next_performance(metric="acc")
    assert isinstance(performance, float)


def test_forecast_next_performance_with_no_evaluations() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_trigger(10, 1, {"acc": 0.8})
    tracker.inform_trigger(10, 1, {"acc": 0.85})
    performance = tracker.forecast_next_performance(metric="acc")
    assert isinstance(performance, float)
    assert performance == 0.85  # Should return the last trigger evaluation


def test_accuracy_forecasts() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=2)
    tracker.inform_evaluation(10, 1, {"acc": 0})
    tracker.inform_evaluation(10, 2, {"acc": 0})
    tracker.inform_evaluation(10, 3, {"acc": 0})
    tracker.inform_evaluation(10, 4, {"acc": 0})
    tracker.inform_evaluation(10, 5, {"acc": 0})
    performance = tracker.forecast_next_accuracy(method="rolling_average")
    assert performance == (9 + 8 + 7 + 6 + 5) / 5 / 10

    tracker.inform_trigger(10, 1, {"acc": 0})
    tracker.inform_trigger(10, 2, {"acc": 0})
    tracker.inform_trigger(10, 3, {"acc": 0})
    tracker.inform_trigger(10, 4, {"acc": 0})
    tracker.inform_trigger(10, 5, {"acc": 0})
    performance = tracker.forecast_expected_accuracy("rolling_average")  # window size 2
    assert tracker.forecast_expected_accuracy("rolling_average") == (6 + 5) / 2 / 10
