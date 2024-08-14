from collections import deque

from sklearn import linear_model

from modyn.const.types import ForecastingMethod


class DataDensityTracker:
    """Observes a stream of data chunks and estimates the time density of the
    data.

    Assumes that the data chunks are ordered by time. For the first
    chunks only number of samples, start and end time of a batch is
    considered. Starting with the second batch, the time between the
    last sample of the previous batch and the first sample of the
    current batch is considered as well.

    Most use cases have constant batch sizes.
    """

    def __init__(self, window_size: int) -> None:
        """
        Args:
            window_size: How many batches the memory for the rolling average should hold.
        """
        self.batch_memory: deque[tuple[int, int]] = deque(maxlen=window_size)
        """Memory of the last `window_size` batches containing the number of
        samples and the time range of the batch in seconds."""

        self.previous_batch_end_time: int | None = None

    def inform_data(self, data: list[tuple[int, int]]) -> None:
        """Informs the tracker about new data batch."""
        if len(data) == 0:
            return

        num_seconds = (
            data[-1][1] - self.previous_batch_end_time
            if self.previous_batch_end_time is not None
            else data[-1][1] - data[0][1]
        )

        self.batch_memory.append((len(data), num_seconds))
        self.previous_batch_end_time = data[-1][1]

    def previous_batch_samples(self) -> int:
        """Returns the number of samples in the last batch."""
        assert len(self.batch_memory) > 0, "No data in memory, calibration needed."
        return self.batch_memory[-1][0]

    def needs_calibration(self) -> bool:
        """Checks if the tracker has enough data for a forecast."""
        return len(self.batch_memory) == 0

    def forecast_density(self, method: ForecastingMethod = "ridge_regression") -> float:
        """Forecasts the data density based on the current memory.

        Returns:
            The forecasted data density as ratio of samples per second.
        """

        assert len(self.batch_memory) > 0, "No data in memory, calibration needed."

        ratio_series = [num_samples / num_seconds for num_samples, num_seconds in self.batch_memory]

        if len(self.batch_memory) < 5 or method == "rolling_average":
            return sum(ratio_series) / len(ratio_series)

        # Ridge regression estimator for scalar time series forecasting
        reg = linear_model.Ridge(alpha=0.5)
        reg.fit([[i] for i in range(len(ratio_series))], ratio_series)
        return reg.predict([[len(ratio_series)]])[0]
