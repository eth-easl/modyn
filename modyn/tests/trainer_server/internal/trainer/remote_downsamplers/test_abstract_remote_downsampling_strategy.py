# pylint: disable=abstract-class-instantiated
from unittest.mock import patch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_batch_then_sample_general():
    downsampling_ratio = 50

    params_from_selector = {"downsampling_ratio": downsampling_ratio}
    sampler = AbstractRemoteDownsamplingStrategy(154, 128, 64, params_from_selector)

    assert hasattr(sampler, "downsampling_ratio")
    assert sampler.downsampling_ratio == 50
    assert sampler.trigger_id == 128
    assert sampler.pipeline_id == 154
    assert sampler.batch_size == 64
