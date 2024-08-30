class IncorporationLatencyTracker:
    """Tracker for latency-based regret metrics like data-incorporation-
    latency."""

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

    def add_latency(self, regret: float, batch_duration: float) -> float:
        """Add the new regret from the last interval to the cumulative regret.

        Applicable if the regret can uniformly be aggregated to a scalar for every reporting data
        batch. (e.g. batch drift distance)

        This assumes uniformity in the aggregated regret value. If the regret is not uniform.
        E.g. when it's build from the per-sample regrets of samples with varying timestamps, we
        cannot simply assume that every regret component can be weighted by the period duration when adding it to
        the cumulative regret. Some samples might only have arrived at the end of the period and thus
        need a smaller weight. Consider using `add_latencies`.

        Args:
            regret_sum: The new regret value at the end of the interval.
            interval_duration: The duration of the last period in seconds

        Returns:
            Most recent cumulative regret value.
        """

        # newly arrived `regret` has existed for `batch_duration / 2` seconds on average;
        # old regret persists for the entire `batch_duration`
        self._cumulative_latency_regret += self._current_regret * batch_duration + regret * (batch_duration / 2.0)
        self._current_regret += regret

        return self._cumulative_latency_regret

    def add_latencies(
        self,
        regrets: list[tuple[int, float]],
        start_timestamp: int,
        batch_duration: float,
    ) -> float:
        """Add the new regret after computing the regret sum from a list of
        per-sample regrets.

        Addressed the non-uniformity in the regret values by computing the regret sum from the per-sample.

        Args:
            regrets: List of regrets for each sample in the last period with their timestamps.
            start_timestamp: The timestamp of the start of the last period.
            end_timestamp: The timestamp of the end of the last period.
            batch_duration: The duration of the last period in seconds.

        Returns:
            Most recent cumulative regret value.
        """
        # We split the regrets into two parts: those that arrived in the last period and those that have been
        # around for longer. In the latency cumulation (area under curve), the first make up a triangular shape,
        # while the second contribute to the rectangle.

        end_timestamp = start_timestamp + batch_duration
        regrets_durations = [(end_timestamp - timestamp, regret) for timestamp, regret in regrets]
        new_regret = sum(regret for _, regret in regrets)

        # rectangular shape of the area under the curve for the recently arrived regrets
        new_regret_latency = sum(duration * regret for duration, regret in regrets_durations)

        # old regret units that still contribute to area under curve (rectangular shape)
        old_regret_latency = self._current_regret * batch_duration

        self._current_regret += new_regret
        self._cumulative_latency_regret += old_regret_latency + new_regret_latency

        return self._cumulative_latency_regret

    def inform_trigger(self) -> None:
        """Informs the tracker about a trigger which will reset the counter."""
        self._current_regret = 0
        self._cumulative_latency_regret = 0
