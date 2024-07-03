import numpy as np
import pandas as pd
from modyn.supervisor.internal.triggers.drift.metrics.mmd import mmd2_two_sample_test


def _add_col_prefixes(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df.columns = [f"{prefix}{col}" for col in df.columns]
    return df


def test_modyn_mmd_metric(data_ref: np.ndarray, data_h0: np.ndarray, data_cur: np.ndarray) -> None:
    # on h0
    results = mmd2_two_sample_test(
        reference_emb=data_ref,
        current_emb=data_h0,
        num_bootstraps=200,
        quantile_probability=0.05,
        num_workers=4,
    )
    assert not results.is_drift
    assert 0.94 <= results.p_val < 1

    # on current data
    results = mmd2_two_sample_test(
        reference_emb=data_ref,
        current_emb=data_cur,
        num_bootstraps=200,
        quantile_probability=0.05,
        num_workers=4,
    )
    assert results.is_drift
    assert 0 <= results.distance <= 0.3
