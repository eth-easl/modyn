from modyn.supervisor.internal.triggers.cost.incorporation_latency_tracker import (
    IncorporationLatencyTracker,
)


def test_incorporation_latency_tracker() -> None:
    tracker = IncorporationLatencyTracker()
    assert tracker.cumulative_latency_regret == 0.0

    cumulative_regret1 = tracker.add_latency(1.0, 2.0)
    assert tracker._current_regret == 1.0
    assert cumulative_regret1 == tracker.cumulative_latency_regret == 0.0 + 0.5 * 2.0

    cumulative_regret2 = tracker.add_latency(0.5, 4.0)
    assert tracker._current_regret == 1.5
    assert cumulative_regret2 == tracker.cumulative_latency_regret == cumulative_regret1 + 1.0 * 4.0 + 0.5 * 4.0 * 0.5

    tracker.inform_trigger()
    assert tracker.cumulative_latency_regret == 0.0

    cumulative_regret3 = tracker.add_latency(2.0, 1.0)
    assert tracker._current_regret == 2.0
    assert cumulative_regret3 == tracker.cumulative_latency_regret == 2.0 * 1.0 * 0.5


def test_add_latencies() -> None:
    tracker = IncorporationLatencyTracker()
    assert tracker.cumulative_latency_regret == 0.0

    cumulative_regret_1 = tracker.add_latencies([(1, 1.0), (2, 0.5)], 0, 3.0)
    assert tracker._current_regret == 1.5
    assert cumulative_regret_1 == tracker.cumulative_latency_regret == (1.0 * (3.0 - 1.0) + 0.5 * (3.0 - 2.0))

    # period: time=3 to 13
    cumulative_regret_2 = tracker.add_latencies([(6, 4.0), (7, 2.0)], 3, 10.0)
    assert tracker._current_regret == 1.5 + (4.0 + 2.0)
    assert (
        cumulative_regret_2
        == tracker.cumulative_latency_regret
        == cumulative_regret_1 + 1.5 * 10.0 + (4.0 * (13.0 - 6.0) + 2.0 * (13.0 - 7.0))
    )
