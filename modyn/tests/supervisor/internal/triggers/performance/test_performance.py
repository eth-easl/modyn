import pytest

from modyn.supervisor.internal.triggers.performance.performance import (
    PerformanceTracker,
)


def test_initial_state() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    assert len(tracker.trigger_evaluation_memory) == 0
    assert len(tracker.since_last_trigger) == 0


def test_inform_evaluation() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_evaluation(0.8)
    assert len(tracker.since_last_trigger) == 1
    assert tracker.since_last_trigger[0] == 0.8


def test_inform_trigger() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_evaluation(0.8)
    tracker.inform_trigger(0.9)
    assert len(tracker.trigger_evaluation_memory) == 1
    assert tracker.trigger_evaluation_memory[-1] == 0.9
    assert len(tracker.since_last_trigger) == 0  # Reset after trigger


def test_inform_trigger_memory_rollover() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_trigger(0.8)
    tracker.inform_trigger(0.85)
    tracker.inform_trigger(0.9)
    tracker.inform_trigger(0.95)  # This should push out the first evaluation
    assert len(tracker.trigger_evaluation_memory) == 3
    assert tracker.trigger_evaluation_memory[0] == 0.85  # First entry should be the second trigger


def test_forecast_expected_performance_no_trigger() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    with pytest.raises(AssertionError):
        tracker.forecast_expected_performance()


def test_forecast_expected_performance_simple_average() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_trigger(0.8)
    tracker.inform_trigger(0.85)
    performance = tracker.forecast_expected_performance("lookahead")
    assert performance == 0.825  # Simple average of 0.8 and 0.85


def test_forecast_expected_performance_ridge_regression() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=5)
    tracker.inform_trigger(0.8)
    tracker.inform_trigger(0.85)
    tracker.inform_trigger(0.9)
    tracker.inform_trigger(0.95)
    tracker.inform_trigger(1.0)
    performance = tracker.forecast_expected_performance("lookahead")
    assert isinstance(performance, float)  # Should run ridge regression


def test_forecast_next_performance_no_trigger() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    with pytest.raises(AssertionError):
        tracker.forecast_next_performance()


def test_forecast_next_performance_simple_average() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_evaluation(0.8)
    tracker.inform_evaluation(0.85)
    performance = tracker.forecast_next_performance("lookahead")
    assert performance == 0.825  # Simple average of 0.8 and 0.85


def test_forecast_next_performance_ridge_regression() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=5)
    tracker.inform_evaluation(0.8)
    tracker.inform_evaluation(0.85)
    tracker.inform_evaluation(0.9)
    tracker.inform_evaluation(0.95)
    tracker.inform_evaluation(1.0)
    performance = tracker.forecast_next_performance()
    assert isinstance(performance, float)  # Should run ridge regression


def test_forecast_next_performance_with_no_evaluations() -> None:
    tracker = PerformanceTracker(trigger_eval_window_size=3)
    tracker.inform_trigger(0.8)
    tracker.inform_trigger(0.85)
    performance = tracker.forecast_next_performance()
    assert performance == 0.85  # Should return the last trigger evaluation
