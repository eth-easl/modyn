from collections import deque

import numpy as np
from sklearn import linear_model


class DataDensityTracker:
    """Observes a stream trigger costs (wall clack time measurements) and
    maintains a linear model assuming a linear relationship between the number
    of samples and the time it takes to process them."""

    def __init__(self, window_size: int = 1000) -> None:
        """
        Args:
            window_size: How many trigger into the past should be considered for the linear model.
        """
        self.measurements: deque[tuple[int, float]] = deque(maxlen=window_size)
        """List of measurements of number of samples and the resulted training
        time for the last `window_size` triggers."""

        self._linear_model = linear_model.Ridge()

    def inform_trigger(self, number_samples: int, elapsed_time: float) -> None:
        """Informs the tracker about new data batch."""
        self.measurements.append((number_samples, elapsed_time))

        samples = np.array([x[0] for x in self.measurements]).reshape(-1, 1)
        observations = [x[1] for x in self.measurements]

        self._linear_model.fit(samples, observations)

    def needs_calibration(self) -> bool:
        """Checks if the tracker has enough data for a forecast.

        After one trigger inform, the tracker forecasts with a constant
        model. With the second trigger the model can learn the
        y-intercept and start making meaningful forecasts.
        """
        return len(self.measurements) == 0 or not hasattr(self._linear_model, "coef_")

    def forecast_training_time(self, number_samples: int) -> float:
        """Forecasts the training time for a given number of samples."""
        assert not self.needs_calibration(), "The tracker needs more data to make a forecast."
        return self._linear_model.predict(np.array([[number_samples]]))[0]
