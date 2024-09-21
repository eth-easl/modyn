import logging

from modyn.const.types import ForecastingMethod
from modyn.supervisor.internal.triggers.performance.data_density_tracker import (
    DataDensityTracker,
)
from modyn.supervisor.internal.triggers.performance.performance_tracker import (
    PerformanceTracker,
)


class NumberAvoidableMisclassificationEstimator:
    """Immutable class offering estimation functionality for the number of
    avoidable misclassifications.

    Used in `StaticNumberAvoidableMisclassificationDecisionPolicy` to make decisions based on the cumulated
    avoidable misclassifications.
    """

    def __init__(self, expected_accuracy: float | None = None, allow_reduction: bool = False):
        """
        Args:
            threshold: The threshold of cumulated avoidable misclassifications.
        """

        self.expected_accuracy = expected_accuracy
        self.allow_reduction = allow_reduction

    def estimate_avoidable_misclassifications(
        self,
        update_interval_samples: int,
        data_density: DataDensityTracker,
        performance_tracker: PerformanceTracker,
        method: ForecastingMethod,
    ) -> tuple[float, float]:
        """Utilizes the state of `DataDensityTracker` and `PerformanceTracker`
        to estimate the number of avoidable misclassifications.

        We support both the "hindsight" and "forecast" mode.

        In the "hindsight" mode, the decision is made based on the current performance and the cumulated avoidable misclassifications.

        Returns:
            Tuple of the number of avoidable misclassifications and the forecasted number of avoidable misclassifications.
        """

        # --------------------------------- compute new_avoidable_misclassifications --------------------------------- #
        # compute the number of avoidable misclassifications by retrieving the actual misclassifications
        # and the expected misclassifications through the expected accuracy for the last interval.
        previous_interval_num_misclassifications = performance_tracker.previous_batch_num_misclassifications

        if self.expected_accuracy is None:
            logging.warning(
                "Forecasting mode is not supported yet, it requires tracking the performance right after trigger. "
                "However, after triggers the models has learned from the last detection interval. We would need to "
                "maintain a holdout set for this."
            )

        # the expected performance won't change unless there's a trigger
        expected_accuracy = (
            performance_tracker.forecast_expected_accuracy(method=method)
            if self.expected_accuracy is None
            else self.expected_accuracy
        )
        previous_expected_num_misclassifications = (1 - expected_accuracy) * data_density.previous_batch_num_samples
        new_avoidable_misclassifications = (
            previous_interval_num_misclassifications - previous_expected_num_misclassifications
        )
        if new_avoidable_misclassifications < 0 and not self.allow_reduction:
            new_avoidable_misclassifications = 0

        # --------------------------------- compute new_avoidable_misclassifications --------------------------------- #
        # forecasted_data_density = data_density.forecast_density(method=method)
        forecast_accuracy = performance_tracker.forecast_next_accuracy(method=method)

        accuracy_delta = expected_accuracy - forecast_accuracy
        if accuracy_delta < 0 and not self.allow_reduction:
            accuracy_delta = 0

        # new misclassification = accuracy * samples; samples = data_density * interval_duration
        forecast_new_avoidable_misclassifications = accuracy_delta * update_interval_samples

        return (
            new_avoidable_misclassifications,
            forecast_new_avoidable_misclassifications,
        )
