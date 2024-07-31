import statistics

import numpy as np
import pytest

from modyn.config.schema.pipeline import AlibiDetectCVMDriftMetric, AlibiDetectKSDriftMetric, AlibiDetectMmdDriftMetric
from modyn.supervisor.internal.triggers.drift.alibi_detector import AlibiDriftDetector


@pytest.fixture
def mmd_drift_metric() -> AlibiDetectMmdDriftMetric:
    return AlibiDetectMmdDriftMetric(
        p_val=0.05,
        device=None,
        num_permutations=100,
        kernel="GaussianRBF",
    )


@pytest.fixture
def ks_drift_metric() -> AlibiDetectKSDriftMetric:
    return AlibiDetectKSDriftMetric(
        p_val=0.05,
        correction="bonferroni",
        alternative_hypothesis="two-sided",
    )


@pytest.fixture
def cvm_drift_metric() -> AlibiDetectCVMDriftMetric:
    return AlibiDetectCVMDriftMetric(
        p_val=0.05,
        correction="bonferroni",
    )


def test_alibi_detect_drift_metric(
    mmd_drift_metric: AlibiDetectMmdDriftMetric,
    ks_drift_metric: AlibiDetectKSDriftMetric,
    cvm_drift_metric: AlibiDetectCVMDriftMetric,
    data_ref: np.ndarray,
    data_h0: np.ndarray,
    data_cur: np.ndarray,
) -> None:
    detector = [
        ("mmd", AlibiDriftDetector({"mmd": mmd_drift_metric})),
        ("ks", AlibiDriftDetector({"ks": ks_drift_metric})),
        ("cvm", AlibiDriftDetector({"cvm": cvm_drift_metric})),
    ]
    for name, ad in detector:
        assert isinstance(ad, AlibiDriftDetector)

        # on h0
        results = ad.detect_drift(data_ref, data_h0)
        assert not results[name].is_drift

        # on current data
        results = ad.detect_drift(data_ref, data_cur)
        assert results[name].is_drift
        assert (
            0
            <= statistics.mean(
                [results[name].distance] if not isinstance(results[name].distance, list) else results[name].distance
            )
            < (1.0 if name == "cvm" else 0.2)
        )
        assert (
            0
            <= statistics.mean(
                [results[name].p_val] if not isinstance(results[name].p_val, list) else results[name].p_val
            )
            < 0.015
        )
        assert results[name].threshold is not None
        assert results[name].metric_id == name

    ad = AlibiDriftDetector(
        {
            "mmd": mmd_drift_metric,
            "ks": ks_drift_metric,
            "cvm": cvm_drift_metric,
        }
    )
    results = ad.detect_drift(data_ref, data_cur)
    assert len(results) == 3
