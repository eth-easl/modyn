from collections import deque

import numpy as np
import pytest

from modyn.supervisor.internal.triggers.cost.cost_tracker import CostTracker


@pytest.fixture
def tracker() -> CostTracker:
    return CostTracker(window_size=3)


def test_initial_state_and_needs_calibration(tracker: CostTracker) -> None:
    assert len(tracker.measurements) == 0
    assert not tracker.needs_calibration()

    tracker.inform_trigger(10, 0.5)
    assert tracker.needs_calibration()


def test_inform_trigger_updates_measurements(tracker: CostTracker) -> None:
    tracker.inform_trigger(10, 0.5)
    tracker.inform_trigger(20, 1.0)
    assert tracker.measurements == deque([(10, 0.5), (20, 1.0)], maxlen=3)

    tracker.inform_trigger(30, 1.5)
    assert tracker.measurements == deque([(10, 0.5), (20, 1.0), (30, 1.5)], maxlen=3)

    tracker.inform_trigger(40, 2.0)
    assert tracker.measurements == deque([(20, 1.0), (30, 1.5), (40, 2.0)], maxlen=3)


def test_forecast_training_time_and_model_retraining(
    tracker: CostTracker,
) -> None:
    # Inform with data points
    with pytest.raises(AssertionError, match="The tracker needs more data to make a forecast."):
        tracker.forecast_training_time(20)

    tracker.inform_trigger(10, 0.5)
    assert tracker.forecast_training_time(20) == 0.5  # first inform will only configure a constant model
    assert tracker.forecast_training_time(0) == 0.5

    tracker.inform_trigger(20, 1.0)
    initial_coef = tracker._linear_model.coef_.copy()

    tracker.inform_trigger(30, 1.5)
    updated_coef = tracker._linear_model.coef_.copy()
    assert not np.array_equal(initial_coef, updated_coef)
