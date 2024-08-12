from typing import Deque, Literal

from sklearn import linear_model


class PerformanceTracker:
    """Observes a stream of performance evaluation and estimates performance on the next chunk.

    While no trigger happens, the estimated performances is calculated from the series of evaluations
    after every of the last n-triggers. The next observed performance is also forecasted from the series
    of evaluations since the last trigger. When a trigger happens, this series of observations evaluations is reset.
    """

    def __init__(self, trigger_eval_window_size: int) -> None:
        """
        Args:
            window_size: How many evaluations after triggers should be kept in memory.
        """
        self.trigger_evaluation_memory: Deque[float] = Deque(maxlen=trigger_eval_window_size)
        """Memory of the last `window_size` evaluations after triggers."""

        self.since_last_trigger: list[float] = list()

    def inform_evaluation(self, evaluation: float) -> None:
        """Informs the tracker about a new evaluation."""
        self.since_last_trigger.append(evaluation)

    def inform_trigger(self, evaluation: float) -> None:
        """Informs the tracker about a new trigger and resets the memory."""
        self.trigger_evaluation_memory.append(evaluation)
        self.since_last_trigger = list()

    def forecast_expected_performance(self, mode: Literal["hindsight", "lookahead"] = "lookahead") -> float:
        """Forecasts the performance based on the current memory.

        Returns:
            The forecasted performance.
        """

        assert len(self.trigger_evaluation_memory) > 0, "No trigger happened yet. Calibration needed."

        if len(self.since_last_trigger) < 5 or mode == "hindsight":
            return sum(self.trigger_evaluation_memory) / len(self.trigger_evaluation_memory)

        # Ridge regression estimator for scalar time series forecasting
        reg = linear_model.Ridge(alpha=0.5)
        reg.fit(
            [[i] for i in range(len(self.trigger_evaluation_memory))],
            self.trigger_evaluation_memory,
        )
        return reg.predict([[len(self.trigger_evaluation_memory)]])[0]

    def forecast_next_performance(self, mode: Literal["hindsight", "lookahead"] = "lookahead") -> float:
        """Forecasts the performance based on the current memory.

        Returns:
            The forecasted (observed) performance.
        """

        if len(self.since_last_trigger) == 0:
            assert self.trigger_evaluation_memory, "No trigger happened yet."
            return self.trigger_evaluation_memory[-1]

        if len(self.since_last_trigger) < 5 or mode == "hindsight":
            return sum(self.since_last_trigger) / len(self.since_last_trigger)

        # Ridge regression estimator for scalar time series forecasting
        reg = linear_model.Ridge(alpha=0.5)
        reg.fit([[i] for i in range(len(self.since_last_trigger))], self.since_last_trigger)
        return reg.predict([[len(self.since_last_trigger)]])[0]
