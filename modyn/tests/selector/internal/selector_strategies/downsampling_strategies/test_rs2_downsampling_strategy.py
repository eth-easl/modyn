import os
import pathlib
import tempfile

from modyn.config.schema.sampling.downsampling_config import RS2DownsamplingConfig
from modyn.selector.internal.selector_strategies.downsampling_strategies import RS2DownsamplingStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
TMP_DIR = tempfile.mkdtemp()


def test_init_rs2():
    # Test init works
    strat = RS2DownsamplingStrategy(
        RS2DownsamplingConfig(ratio=10, with_replacement=True),
        {},
        0,
        1000,
    )

    assert strat.downsampling_ratio == 10
    assert strat.requires_remote_computation
    assert strat.maximum_keys_in_memory == 1000

    name = strat.remote_downsampling_strategy_name
    assert isinstance(name, str)
    assert name == "RemoteRS2Downsampling"

    params = strat.downsampling_params
    assert "replacement" in params
    assert params["replacement"]
