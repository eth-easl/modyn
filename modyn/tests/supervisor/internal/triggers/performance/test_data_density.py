import pytest

from modyn.supervisor.internal.triggers.performance.data_density import (
    DataDensityTracker,
)


def test_initial_state() -> None:
    tracker = DataDensityTracker(window_size=3)
    assert len(tracker.batch_memory) == 0
    assert tracker.previous_batch_end_time is None


def test_inform_data_empty_batch() -> None:
    tracker = DataDensityTracker(window_size=3)
    tracker.inform_data([])
    assert len(tracker.batch_memory) == 0  # Should not change anything


def test_inform_data_batches() -> None:
    tracker = DataDensityTracker(window_size=3)
    assert tracker.needs_calibration()
    tracker.inform_data([(1, 10), (2, 20), (3, 30)])
    assert not tracker.needs_calibration()
    assert len(tracker.batch_memory) == 1
    assert tracker.previous_batch_end_time == 30
    assert tracker.batch_memory[-1] == (3, 20)  # (len(data), end_time - start_time)
    tracker.inform_data([(4, 40), (5, 50), (6, 60)])
    assert len(tracker.batch_memory) == 2
    assert tracker.batch_memory[-1] == (
        3,
        30,
    )  # Time gap is 10 due to the interval between batches


def test_inform_data_window_rollover() -> None:
    tracker = DataDensityTracker(window_size=3)
    tracker.inform_data([(1, 10), (2, 20), (3, 30)])
    tracker.inform_data([(4, 40), (5, 50), (6, 60)])
    tracker.inform_data([(7, 70), (8, 80), (9, 90)])
    tracker.inform_data([(10, 100), (11, 110), (12, 120)])  # This should push out the first batch
    assert len(tracker.batch_memory) == 3
    assert tracker.batch_memory[0] == (
        3,
        30,
    )  # First entry in the deque should be the second batch


def test_forecast_density_empty_memory() -> None:
    tracker = DataDensityTracker(window_size=3)
    with pytest.raises(AssertionError):
        tracker.forecast_density()


def test_forecast_density_simple_average() -> None:
    tracker = DataDensityTracker(window_size=3)
    tracker.inform_data([(1, 10), (2, 20), (3, 30)])  # density: 3/20[s]
    tracker.inform_data([(4, 40), (5, 50), (6, 60)])  # density: 3/30[s]
    density = tracker.forecast_density()
    assert density == (3 / 20 + 3 / 30) / 2


def test_forecast_density_ridge_regression() -> None:
    tracker = DataDensityTracker(window_size=5)
    tracker.inform_data([(1, 10), (2, 20), (3, 30)])
    tracker.inform_data([(4, 40), (5, 50), (6, 60)])
    tracker.inform_data([(7, 70), (8, 80), (9, 90)])
    tracker.inform_data([(10, 100), (11, 110), (12, 120)])
    tracker.inform_data([(13, 130), (14, 140), (15, 150)])
    density = tracker.forecast_density("lookahead")
    assert isinstance(density, float)  # Since it should run ridge regression


def test_forecast_density_with_varied_batches() -> None:
    tracker = DataDensityTracker(window_size=5)
    tracker.inform_data([(1, 10), (2, 20)])
    tracker.inform_data([(3, 30), (4, 40), (5, 50)])
    tracker.inform_data([(6, 60), (7, 70)])
    density = tracker.forecast_density("lookahead")
    assert isinstance(density, float)
    assert density > 0  # Ensure a positive density value
