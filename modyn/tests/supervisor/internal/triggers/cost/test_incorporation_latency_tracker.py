from modyn.supervisor.internal.triggers.cost.incorporation_latency_tracker import (
    IncorporationLatencyTracker,
)


def test_incorporation_latency_tracker() -> None:
    tracker = IncorporationLatencyTracker()
    assert tracker.cumulative_latency_regret == 0.0

    cumulative_regret = tracker.add_latency(1.0, 2.0)
    assert tracker._current_regret == 1.0
    assert cumulative_regret == tracker.cumulative_latency_regret == 2.0

    cumulative_regret = tracker.add_latency(0.5, 3.0)
    assert tracker._current_regret == 1.5
    assert cumulative_regret == tracker.cumulative_latency_regret == 2.0 + 1.5 * 3.0

    tracker.inform_trigger()
    assert tracker.cumulative_latency_regret == 0.0

    cumulative_regret = tracker.add_latency(2.0, 1.0)
    assert tracker._current_regret == 2.0
    assert cumulative_regret == tracker.cumulative_latency_regret == 2.0
