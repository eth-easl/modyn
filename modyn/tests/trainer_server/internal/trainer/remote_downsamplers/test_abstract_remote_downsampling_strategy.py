# pylint: disable=abstract-class-instantiated
from unittest.mock import patch

import pytest
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_batch_then_sample_general():
    downsampled_batch_size = 5

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size, "sample_before_batch": False}
    sampler = AbstractRemoteDownsamplingStrategy(0, 0, 0, params_from_selector)

    assert hasattr(sampler, "downsampled_batch_size")
    assert not hasattr(sampler, "sample_then_batch_handler")
    assert not hasattr(sampler, "downsampled_batch_ratio")

    with pytest.raises(AssertionError):
        sampler.get_downsampled_batch_ratio()


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_sample_then_batch_general():
    downsampled_batch_ratio = 50

    with pytest.raises(AssertionError):
        params_from_selector = {"downsampled_batch_size": downsampled_batch_ratio, "sample_before_batch": True}
        sampler = AbstractRemoteDownsamplingStrategy(0, 0, 100, params_from_selector)

    with pytest.raises(AssertionError):
        params_from_selector = {
            "downsampled_batch_ratio": downsampled_batch_ratio,
        }
        sampler = AbstractRemoteDownsamplingStrategy(0, 0, 100, params_from_selector)

    params_from_selector = {
        "downsampled_batch_ratio": downsampled_batch_ratio,
        "sample_before_batch": True,
        "maximum_keys_in_memory": 1000,
    }
    sampler = AbstractRemoteDownsamplingStrategy(0, 0, 100, params_from_selector)

    assert not hasattr(sampler, "downsampled_batch_size")
    assert hasattr(sampler, "downsampled_batch_ratio")

    assert sampler.get_downsampled_batch_ratio() == downsampled_batch_ratio
