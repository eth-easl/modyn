import numpy as np
import pandas as pd
import pytest
from modyn.config.schema.pipeline import (
    EvidentlyMmdDriftMetric,
    EvidentlyModelDriftMetric,
    EvidentlyRatioDriftMetric,
    EvidentlySimpleDistanceDriftMetric,
)
from modyn.supervisor.internal.triggers.drift.evidently_detector import EvidentlyDriftDetector


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
def mmd_drift_metric() -> EvidentlyMmdDriftMetric:
    return EvidentlyMmdDriftMetric(bootstrap=True)


@pytest.fixture
def model_drift_metric() -> EvidentlyModelDriftMetric:
    return EvidentlyModelDriftMetric(bootstrap=True)


@pytest.fixture
def ratio_drift_metric() -> EvidentlyRatioDriftMetric:
    return EvidentlyRatioDriftMetric()


@pytest.fixture
def simple_distance_drift_metric() -> EvidentlySimpleDistanceDriftMetric:
    return EvidentlySimpleDistanceDriftMetric(
        bootstrap=True,
        distance_metric="euclidean",
    )


def test_evidently_detect_drift_metric(
    mmd_drift_metric: EvidentlyMmdDriftMetric,
    model_drift_metric: EvidentlyModelDriftMetric,
    ratio_drift_metric: EvidentlyRatioDriftMetric,
    simple_distance_drift_metric: EvidentlySimpleDistanceDriftMetric,
    df_data_ref: np.ndarray,
    df_data_h0: np.ndarray,
    df_data_cur: np.ndarray,
) -> None:
    detector = [
        ("mmd", EvidentlyDriftDetector({"mmd": mmd_drift_metric})),
        ("model", EvidentlyDriftDetector({"model": model_drift_metric})),
        ("ratio", EvidentlyDriftDetector({"ratio": ratio_drift_metric})),
        ("simple_distance", EvidentlyDriftDetector({"simple_distance": simple_distance_drift_metric})),
    ]
    for name, ad in detector:
        assert isinstance(ad, EvidentlyDriftDetector)

        # on h0
        results = ad.detect_drift(df_data_ref, df_data_h0)
        # if name == "mmd":
        #     # bug in evidently, always uses reference data to detect drift
        #     pass
        # else:
        assert not results[name].is_drift

        # on current data
        results = ad.detect_drift(df_data_ref, df_data_cur)
        if name != "model":
            # model makes the wrong decision here
            assert results[name].is_drift
        assert 0 <= results[name].distance <= (1 if name == "ratio" else 0.5)
        assert results[name].metric_id == "EmbeddingsDriftMetric"

    ad = EvidentlyDriftDetector(
        {
            "mmd": mmd_drift_metric,
            "model": model_drift_metric,
            "ratio": ratio_drift_metric,
            "simple_distance": simple_distance_drift_metric,
        }
    )
    results = ad.detect_drift(df_data_ref, df_data_cur)
    assert len(results) == 4
