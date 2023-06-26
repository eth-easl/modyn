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
            "downsampled_batch_size": 10,
        }
    )

    assert strat.downsampled_batch_size == 10
    assert isinstance(strat.get_downsampling_strategy(), str)
    assert strat.get_requires_remote_computation()


def test_command_loss():
    # Test init works
    strat = LossDownsamplingStrategy(
        {
            "limit": -1,
            "reset_after_trigger": False,
            "presampling_ratio": 80,
            "downsampled_batch_size": 10,
        }
    )

    name = strat.get_downsampling_strategy()
    params = strat.get_downsampling_params()
    assert isinstance(name, str)
    assert name == "RemoteLossDownsampling"
    assert "downsampled_batch_size" in params
    assert params["downsampled_batch_size"] == 10
    assert strat.get_requires_remote_computation()
