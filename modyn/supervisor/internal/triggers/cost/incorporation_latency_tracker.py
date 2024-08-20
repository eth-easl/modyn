class IncorporationLatencyTracker:
    def __init__(self) -> None:
        self._current_regret = 0.0
        """The current value of the regret metric at the end of the last
        interval."""

        self._cumulative_latency_regret = 0.0
        """Cumulated regret (latency) reflecting the area under the regret
        curve."""

    @property
    def cumulative_latency_regret(self) -> float:
        return self._cumulative_latency_regret

    def add_latency(self, regret: float, period_duration: float) -> float:
        """Add the new regret from the last interval to the cumulative regret.

        Args:
            regret: The regret value at the end of the interval.
            interval_duration: The duration of the last period in seconds

        Returns:
            Most recent cumulative regret value.
        """
        self._current_regret += regret
        self._cumulative_latency_regret += self._current_regret * period_duration

        return self._cumulative_latency_regret

    def inform_trigger(self) -> None:
        """Informs the tracker about a trigger which will reset the counter."""
        self._current_regret = 0
        self._cumulative_latency_regret = 0
