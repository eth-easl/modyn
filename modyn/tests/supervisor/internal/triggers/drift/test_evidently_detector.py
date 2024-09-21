import numpy as np
import pandas as pd
import pytest

from modyn.config.schema.pipeline import (
    EvidentlyModelDriftMetric,
    EvidentlyRatioDriftMetric,
    EvidentlySimpleDistanceDriftMetric,
)
from modyn.config.schema.pipeline.trigger.drift.criterion import (
    DynamicQuantileThresholdCriterion,
)
from modyn.config.schema.pipeline.trigger.drift.evidently import (
    EvidentlyHellingerDistanceDriftMetric,
)
from modyn.supervisor.internal.triggers.drift.detector.evidently import (
    EvidentlyDriftDetector,
    _evidently_additional_metric_computation,
)


def _add_col_prefixes(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df.columns = [f"{prefix}{col}" for col in df.columns]
    return df


@pytest.fixture(scope="module")
def df_data_ref(data_ref: np.ndarray) -> pd.DataFrame:
    return _add_col_prefixes(pd.DataFrame(data_ref), "col_")


@pytest.fixture(scope="module")
def df_data_h0(data_h0: np.ndarray) -> pd.DataFrame:
    return _add_col_prefixes(pd.DataFrame(data_h0), "col_")


@pytest.fixture(scope="module")
def df_data_cur(data_cur: np.ndarray) -> pd.DataFrame:
    return _add_col_prefixes(pd.DataFrame(data_cur), "col_")


@pytest.fixture
def model_drift_metric() -> EvidentlyModelDriftMetric:
    return EvidentlyModelDriftMetric(bootstrap=False, decision_criterion=DynamicQuantileThresholdCriterion())


@pytest.fixture
def ratio_drift_metric() -> EvidentlyRatioDriftMetric:
    return EvidentlyRatioDriftMetric(decision_criterion=DynamicQuantileThresholdCriterion())


@pytest.fixture
def simple_distance_drift_metric() -> EvidentlySimpleDistanceDriftMetric:
    return EvidentlySimpleDistanceDriftMetric(
        bootstrap=False,
        distance_metric="euclidean",
        decision_criterion=DynamicQuantileThresholdCriterion(),
    )


@pytest.fixture
def hellinger_distance_drift_metric() -> EvidentlySimpleDistanceDriftMetric:
    return EvidentlyHellingerDistanceDriftMetric(decision_criterion=DynamicQuantileThresholdCriterion())


def test_evidently_additional_metric_computation_hellinger(
    hellinger_distance_drift_metric: EvidentlyHellingerDistanceDriftMetric,
    df_data_ref: np.ndarray,
    df_data_h0: np.ndarray,
    df_data_cur: np.ndarray,
) -> None:
    metrics = {"hellinger": hellinger_distance_drift_metric}
    results_h0 = _evidently_additional_metric_computation(metrics, df_data_ref, df_data_h0)
    assert len(results_h0) == 1

    results_cur = _evidently_additional_metric_computation(metrics, df_data_ref, df_data_cur)
    assert len(results_cur) == 1

    assert results_h0["hellinger"].distance < 1.25 * results_cur["hellinger"].distance

    # Test strong drift in 1 of 3 coluns
    df_ref = pd.DataFrame(
        {
            "Column1": np.random.normal(loc=0, scale=1, size=1000),
            "Column2": np.random.uniform(low=0, high=1, size=1000),
            "Column3": np.random.exponential(scale=1, size=1000),
        }
    )

    df_h0 = pd.DataFrame(
        {
            "Column1": np.random.normal(loc=0, scale=1, size=500),
            "Column2": np.random.uniform(low=0, high=1, size=500),
            "Column3": np.random.exponential(scale=1, size=500),
        }
    )

    # Create the second dataframe with slightly different distributions
    df_cur = pd.DataFrame(
        {
            "Column1": np.random.normal(loc=0, scale=1, size=500),
            "Column2": np.random.uniform(low=0.5, high=1.5, size=500),
            "Column3": np.random.exponential(scale=1, size=500),
        }
    )
    results = _evidently_additional_metric_computation(metrics, df_ref, df_h0)
    assert results["hellinger"].distance < 0.1
    results = _evidently_additional_metric_computation(metrics, df_ref, df_cur)
    # assuming 2 columns don't drift and 1 does, we expect sth. around (0 + 0 + 1) / 3 = 0.33
    assert 0.2 < results["hellinger"].distance < 1 / 3.0


def test_evidently_detect_drift_metric(
    model_drift_metric: EvidentlyModelDriftMetric,
    ratio_drift_metric: EvidentlyRatioDriftMetric,
    simple_distance_drift_metric: EvidentlySimpleDistanceDriftMetric,
    hellinger_distance_drift_metric: EvidentlyHellingerDistanceDriftMetric,
    df_data_ref: np.ndarray,
    df_data_h0: np.ndarray,
    df_data_cur: np.ndarray,
) -> None:
    detector = [
        ("model", EvidentlyDriftDetector({"model": model_drift_metric})),
        ("ratio", EvidentlyDriftDetector({"ratio": ratio_drift_metric})),
        (
            "simple_distance",
            EvidentlyDriftDetector({"simple_distance": simple_distance_drift_metric}),
        ),
    ]
    for name, ad in detector:
        assert isinstance(ad, EvidentlyDriftDetector)

        # on h0
        results = ad.detect_drift(df_data_ref, df_data_h0, False)
        assert not results[name].is_drift

        # on current data
        results = ad.detect_drift(df_data_ref, df_data_cur, False)
        if name != "model":
            # model makes the wrong decision here
            assert results[name].is_drift
        assert 0 <= results[name].distance <= (1 if name == "ratio" else 0.75)
        assert results[name].metric_id == "EmbeddingsDriftMetric"

    ad = EvidentlyDriftDetector(
        {
            "model": model_drift_metric,
            "ratio": ratio_drift_metric,
            "simple_distance": simple_distance_drift_metric,
            "hellinger": hellinger_distance_drift_metric,
        }
    )
    results = ad.detect_drift(df_data_ref, df_data_cur, False)
    assert len(results) == 4
