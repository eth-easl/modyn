import os
import pathlib
import tempfile

from modyn.config.schema.pipeline.sampling.downsampling_config import UncertaintyDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import UncertaintyDownsamplingStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
TMP_DIR = tempfile.mkdtemp()


def test_init_uncert():
    # Test init works
    strat = UncertaintyDownsamplingStrategy(
        UncertaintyDownsamplingConfig(ratio=10, score_metric="Margin"),
        {},
        0,
        1000,
    )

    assert strat.downsampling_ratio == 10
    assert strat.requires_remote_computation
    assert strat.maximum_keys_in_memory == 1000

    name = strat.remote_downsampling_strategy_name
    assert isinstance(name, str)
    assert name == "RemoteUncertaintyDownsamplingStrategy"

    params = strat.downsampling_params
    assert "score_metric" in params
    assert params["score_metric"] == "Margin"
