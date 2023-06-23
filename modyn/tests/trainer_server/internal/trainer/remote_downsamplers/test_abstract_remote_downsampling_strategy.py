# pylint: disable=abstract-class-instantiated
from unittest.mock import patch

from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_batch_then_sample_general():
    downsampling_ratio = 50

    params_from_selector = {"downsampling_ratio": downsampling_ratio, "sample_then_batch": False}
    sampler = AbstractRemoteDownsamplingStrategy(0, 0, 0, params_from_selector)

    assert hasattr(sampler, "downsampling_ratio")
    assert not hasattr(sampler, "sample_then_batch_handler")
