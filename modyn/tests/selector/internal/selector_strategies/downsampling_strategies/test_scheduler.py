from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    DownsamplingScheduler,
    LossDownsamplingStrategy,
)
from modyn.utils import DownsamplingMode


def get_configs():
    return [
        {"strategy": "Loss", "sample_then_batch": True, "ratio": 50},
        {"strategy": "GradNorm", "sample_then_batch": False, "ratio": 25},
    ]


def test_init():
    conf = get_configs()
    downs = DownsamplingScheduler(conf, [12], 1000)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, LossDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
