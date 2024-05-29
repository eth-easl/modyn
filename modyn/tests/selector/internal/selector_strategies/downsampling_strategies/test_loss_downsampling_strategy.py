import os
import pathlib
import tempfile

from modyn.config.schema.downsampling_config import LossDownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import LossDownsamplingStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
TMP_DIR = tempfile.mkdtemp()


def test_init_loss():
    # Test init works
    strat = LossDownsamplingStrategy(
        LossDownsamplingConfig(ratio=10, sample_then_batch=True),
        {},
        0,
        1000,
    )

    assert strat.downsampling_ratio == 10
    assert strat.requires_remote_computation
    assert strat.maximum_keys_in_memory == 1000

    name = strat.remote_downsampling_strategy_name
    assert isinstance(name, str)
    assert name == "RemoteLossDownsampling"

    params = strat.downsampling_params
    assert "downsampling_ratio" in params
    assert params["downsampling_ratio"] == 10
