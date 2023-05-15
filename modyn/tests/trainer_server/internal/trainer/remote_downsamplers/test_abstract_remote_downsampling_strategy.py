# pylint: disable=abstract-class-instantiated
from unittest.mock import patch

import pytest
import torch
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.sample_then_batch_handler import SampleThenBatchHandler


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_batch_then_sample_general():
    downsampled_batch_size = 5

    params_from_selector = {"downsampled_batch_size": downsampled_batch_size, "sample_before_batch": False}
    sampler = AbstractRemoteDownsamplingStrategy(0, 0, params_from_selector)

    assert hasattr(sampler, "downsampled_batch_size")
    assert not hasattr(sampler, "sample_then_batch_handler")
    assert not hasattr(sampler, "downsampled_batch_ratio")

    with pytest.raises(AssertionError):
        sampler.get_downsampled_batch_ratio()
    with pytest.raises(AssertionError):
        sampler.get_sample_then_batch_accumulator()
    with pytest.raises(AssertionError):
        sampler.get_samples_for_file(1)


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_sample_then_batch_general():
    downsampled_batch_ratio = 50

    with pytest.raises(AssertionError):
        params_from_selector = {"downsampled_batch_size": downsampled_batch_ratio, "sample_before_batch": True}
        sampler = AbstractRemoteDownsamplingStrategy(0, 100, params_from_selector)

    with pytest.raises(AssertionError):
        params_from_selector = {"downsampled_batch_ratio": downsampled_batch_ratio, "sample_before_batch": True}
        sampler = AbstractRemoteDownsamplingStrategy(0, 100, params_from_selector)

    params_from_selector = {
        "downsampled_batch_ratio": downsampled_batch_ratio,
        "sample_before_batch": True,
        "maximum_keys_in_memory": 1000,
    }
    sampler = AbstractRemoteDownsamplingStrategy(0, 100, params_from_selector)

    assert not hasattr(sampler, "downsampled_batch_size")
    assert hasattr(sampler, "sample_then_batch_handler")
    assert hasattr(sampler, "downsampled_batch_ratio")

    assert sampler.get_downsampled_batch_ratio() == downsampled_batch_ratio
    with pytest.raises(AssertionError):
        sampler.batch_then_sample(torch.Tensor([12, 12]), torch.Tensor([1, 1]))


@patch.multiple(AbstractRemoteDownsamplingStrategy, __abstractmethods__=set())
def test_sample_then_batch():
    downsampled_batch_ratio = 50

    params_from_selector = {
        "downsampled_batch_ratio": downsampled_batch_ratio,
        "sample_before_batch": True,
        "maximum_keys_in_memory": 100,
    }
    sampler = AbstractRemoteDownsamplingStrategy(0, 100, params_from_selector)

    handler = sampler.get_sample_then_batch_accumulator()

    assert isinstance(handler, SampleThenBatchHandler)

    with handler:
        handler.accumulate([1, 2, 3, 4], torch.Tensor([1, 2, 3, 4]))
        handler.accumulate([11, 12, 13, 14], torch.Tensor([1, 2, 3, 4]))
        handler.accumulate([21, 22, 23, 24], torch.Tensor([1, 2, 3, 4]))
        handler.accumulate([31, 32, 33, 34], torch.Tensor([1, 2, 3, 4]))

    assert sampler.get_downsampled_batch_ratio() == 50
    assert len(sampler.get_samples_for_file(0)) == 8

    with pytest.raises(AssertionError):
        sampler.get_samples_for_file(1)
