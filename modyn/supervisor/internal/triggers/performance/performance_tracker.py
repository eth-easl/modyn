from collections import deque

from modyn.const.types import ForecastingMethod
from modyn.supervisor.internal.utils.forecast import forecast_next_performance


class PerformanceTracker:
    """Observes a stream of performance evaluations and estimates performance
    on the next chunk.

    While no trigger happens, the estimated performances is calculated
    from the series of evaluations after every of the last n-triggers.
    The next observed performance is also forecasted from the series of
    evaluations since the last trigger. When a trigger happens, this
    series of observations evaluations is reset.

    Provides both the wrapped performance metrics as well as accuracy
    information.
    """

    def __init__(self, trigger_eval_window_size: int) -> None:
        """
        Args:
            window_size: How many evaluations after triggers should be kept in memory.
        """
        self.trigger_evaluation_memory: deque[tuple[int, int, dict[str, float]]] = deque(
            maxlen=trigger_eval_window_size
        )
        """Memory of the last `window_size` evaluations after triggers with
        their number of samples, misclassifications and evaluation scores for
        different metrics.

        After every trigger, the memory is updated with the new
        evaluation.
        """

        self.since_last_trigger: list[tuple[int, int, dict[str, float]]] = list()
        """Memory of the evaluations since the last trigger with their number
        of samples, misclassifications and evaluation scores for different
        metrics.

        Upon trigger, this memory is reset.
        """

    def inform_evaluation(
        self,
        num_samples: int,
        num_misclassifications: int,
        evaluation_scores: dict[str, float],
    ) -> None:
        """Informs the tracker about a new evaluation."""
        self.since_last_trigger.append((num_samples, num_misclassifications, evaluation_scores))

    def inform_trigger(
        self,
        num_samples: int,
        num_misclassifications: int,
        evaluation_scores: dict[str, float],
    ) -> None:
        """Informs the tracker about a new trigger and resets the memory."""
        self.trigger_evaluation_memory.append((num_samples, num_misclassifications, evaluation_scores))

        # first element in the new series is the performance right after trigger
        self.since_last_trigger = [(num_samples, num_misclassifications, evaluation_scores)]

    def previous_batch_num_misclassifications(self) -> int:
        """Returns the number of misclassifications in the previous batch."""
        return self.since_last_trigger[-1][1]

    def forecast_expected_accuracy(self, method: ForecastingMethod = "ridge_regression") -> float:
        """Forecasts the accuracy based on the current memory of evaluations
        right after triggers.

        Returns:
            The forecasted accuracy.
        """
        return forecast_next_performance(
            observations=[1 - p[1] / p[0] for p in self.trigger_evaluation_memory],
            method=method,
        )

    def forecast_next_accuracy(self, method: ForecastingMethod = "ridge_regression") -> float:
        """Forecasts the next accuracy based on the memory of evaluations since
        the last trigger.

        Returns:
            The forecasted (observed) accuracy.
        """
        return max(
            0,
            min(
                1,
                forecast_next_performance(
                    observations=[1 - p[1] / p[0] for p in self.since_last_trigger],
                    method=method,
                ),
            ),
        )

    def forecast_expected_performance(self, metric: str, method: ForecastingMethod = "ridge_regression") -> float:
        """Forecasts the performance based on the current memory of evaluations
        right after triggers.

        Args:
            metric: The metric to forecast the performance for.
            method: The method to use for forecasting.

        Returns:
            The forecasted performance.
        """
        return forecast_next_performance(
            observations=[p[2][metric] for p in self.trigger_evaluation_memory],
            method=method,
        )

    def forecast_next_performance(self, metric: str, method: ForecastingMethod = "ridge_regression") -> float:
        """Forecasts the next performance based on the memory of evaluations
        since the last trigger.

        Args:
            metric: The metric to forecast the performance for.
            method: The method to use for forecasting.

        Returns:
            The forecasted (observed) performance.
        """
        return forecast_next_performance(observations=[p[2][metric] for p in self.since_last_trigger], method=method)
