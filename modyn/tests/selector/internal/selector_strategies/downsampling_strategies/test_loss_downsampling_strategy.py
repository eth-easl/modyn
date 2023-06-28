import os
import pathlib
import tempfile

from modyn.selector.internal.selector_strategies.downsampling_strategies import LossDownsamplingStrategy

database_path = pathlib.Path(os.path.abspath(__file__)).parent / "test_storage.db"
TMP_DIR = tempfile.mkdtemp()


def test_init_loss():
    # Test init works
    strat = LossDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampling_ratio": 10,
            "sample_then_batch": True,
        },
        1000,
    )

    assert strat.downsampling_ratio == 10
    assert isinstance(strat.get_downsampling_strategy(), str)
    assert strat.get_requires_remote_computation()
    assert strat.maximum_keys_in_memory == 1000


def test_command_loss():
    # Test init works
    strat = LossDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampling_ratio": 10,
            "sample_then_batch": True,
        },
        1000,
    )

    name = strat.get_downsampling_strategy()
    params = strat.get_downsampling_params()
    assert isinstance(name, str)
    assert name == "RemoteLossDownsampling"
    assert "downsampling_ratio" in params
    assert params["downsampling_ratio"] == 10
    assert strat.get_requires_remote_computation()
