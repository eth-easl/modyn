# pylint: disable=abstract-class-instantiated
from unittest.mock import patch

import pytest
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_batch_then_sample_general():
    downsampled_batch_ratio = 50

    params_from_selector = {"downsampled_batch_ratio": downsampled_batch_ratio, "sample_then_batch": False}
    sampler = AbstractRemoteDownsamplingStrategy(0, 0, 0, params_from_selector)

    assert hasattr(sampler, "downsampled_batch_ratio")
    assert not hasattr(sampler, "sample_then_batch_handler")

    with pytest.raises(AssertionError):
        sampler.get_downsampled_batch_ratio()
