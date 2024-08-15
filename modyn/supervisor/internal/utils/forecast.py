"""A simple one-dimensional time series forecasting utility."""

from sklearn import linear_model

from modyn.const.types import ForecastingMethod


def forecast_next_performance(
    observations: list[float],
    method: ForecastingMethod = "ridge_regression",
    min_observations_for_ridge: int = 5,
) -> float:
    """Forecasts the next value based on a series of past observations."""

    assert len(observations) > 0, "No trigger happened yet."

    if len(observations) < min_observations_for_ridge or method == "rolling_average":
        return sum(observations) / len(observations)

    # Ridge regression estimator for scalar time series forecasting
    reg = linear_model.Ridge(alpha=0.5)
    reg.fit([[i] for i in range(len(observations))], observations)
    return reg.predict([[len(observations)]])[0]
