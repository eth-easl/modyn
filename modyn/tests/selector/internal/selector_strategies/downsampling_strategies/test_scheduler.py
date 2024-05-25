from unittest.mock import patch

import pytest
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    DownsamplingScheduler,
    GradNormDownsamplingStrategy,
    LossDownsamplingStrategy,
    NoDownsamplingStrategy,
    instantiate_scheduler, AbstractDownsamplingStrategy,
)
from modyn.utils import DownsamplingMode


def get_configs():
    return [
        {"strategy": "Loss", "sample_then_batch": True, "ratio": 50},
        {"strategy": "GradNorm", "sample_then_batch": False, "ratio": 25},
    ]


def get_configs_triple():
    return [
        {"strategy": "Loss", "sample_then_batch": True, "ratio": 50},
        {"strategy": "GradNorm", "sample_then_batch": False, "ratio": 25},
        {
            "strategy": "No",
        },
    ]


def test_init():
    conf = get_configs()
    downs = DownsamplingScheduler({}, 0, conf, [12], 1000)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, LossDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert downs.current_downsampler.downsampling_ratio == 50
    assert downs.next_threshold == 12


@patch.object(AbstractDownsamplingStrategy, "inform_next_trigger")
def test_switch_downsamplers(mock_inform_next_trigger):
    conf = get_configs()
    downs = DownsamplingScheduler({}, 0, conf, [12], 1000)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, LossDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert downs.current_downsampler.downsampling_ratio == 50
    assert downs.next_threshold == 12

    downs.inform_next_trigger(12)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, GradNormDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE
    assert downs.current_downsampler.downsampling_ratio == 25
    assert downs.next_threshold is None
    mock_inform_next_trigger.assert_called_once_with(12)


def test_switch_functions():
    conf = get_configs()
    downs = DownsamplingScheduler({}, 0, conf, [12], 1000)

    # below the threshold
    for i in range(12):
        downs.inform_next_trigger(i)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_period": 1,
            "downsampling_ratio": 50,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": True,
        }
        assert downs.downsampling_strategy == "RemoteLossDownsampling"
        assert downs.training_status_bar_scale == 50

    # above the threshold
    for i in range(12, 20):
        downs.inform_next_trigger(i)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.downsampling_strategy == "RemoteGradNormDownsampling"
        assert downs.training_status_bar_scale == 100


def test_wrong_number_threshold():
    conf = get_configs_triple()
    # just one threshold
    with pytest.raises(ValueError):
        DownsamplingScheduler({}, 0, conf, [12], 560)

    # three thresholds
    with pytest.raises(ValueError):
        DownsamplingScheduler({}, 0, conf, [12, 13, 14], 560)

    # not sorted
    with pytest.raises(ValueError):
        DownsamplingScheduler({}, 0, conf, [12, 11], 560)

    # double threshold
    with pytest.raises(ValueError):
        DownsamplingScheduler({}, 0, conf, [12, 12], 560)

    # valid one
    DownsamplingScheduler({}, 0, conf, [12, 15], 560)


def test_double_threshold():
    conf = get_configs_triple()
    downs = DownsamplingScheduler({}, 0, conf, [12, 15], 1000)

    # below the first threshold
    for i in range(12):
        downs.inform_next_trigger(i)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_period": 1,
            "downsampling_ratio": 50,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": True,
        }
        assert downs.downsampling_strategy == "RemoteLossDownsampling"
        assert downs.training_status_bar_scale == 50

    # above the first threshold, below the second one
    for i in range(12, 15):
        downs.inform_next_trigger(i)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.downsampling_strategy == "RemoteGradNormDownsampling"
        assert downs.training_status_bar_scale == 100

    # above the last threshold
    for i in range(15, 25):
        downs.inform_next_trigger(i)
        assert not downs.requires_remote_computation
        assert downs.downsampling_params == {}
        assert downs.downsampling_strategy == ""
        assert downs.training_status_bar_scale == 100


def test_wrong_trigger():
    conf = get_configs()
    downs = DownsamplingScheduler({}, 0, conf, [12], 1000)

    for i in range(12):
        downs.inform_next_trigger(i)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_period": 1,
            "downsampling_ratio": 50,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": True,
        }
        assert downs.downsampling_strategy == "RemoteLossDownsampling"
        assert downs.training_status_bar_scale == 50

    # pass the threshold for the first time
    downs.inform_next_trigger(13)
    assert downs.requires_remote_computation

    # then ask again for triggers below the threshold, you still get the second downsampler
    for i in range(12):
        downs.inform_next_trigger(i)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.downsampling_strategy == "RemoteGradNormDownsampling"
        assert downs.training_status_bar_scale == 100


def test_instantiate_scheduler_empty():
    scheduler = instantiate_scheduler({}, {}, 0, 100)
    assert isinstance(scheduler.current_downsampler, NoDownsamplingStrategy)
    assert scheduler.next_threshold is None


def test_instantiate_scheduler_just_one():
    config = {"downsampling_config": {"strategy": "Loss", "sample_then_batch": True, "ratio": 50}}
    scheduler = instantiate_scheduler(config, {}, 0, 100)
    assert isinstance(scheduler.current_downsampler, LossDownsamplingStrategy)
    assert scheduler.next_threshold is None


def test_instantiate_scheduler_list():
    config = {"downsampling_config": {"downsampling_list": get_configs(), "downsampling_thresholds": 7}}
    scheduler = instantiate_scheduler(config, {}, 0, 123)

    assert isinstance(scheduler.current_downsampler, LossDownsamplingStrategy)
    assert scheduler.next_threshold == 7

    scheduler.inform_next_trigger(8)
    assert scheduler.requires_remote_computation

    assert isinstance(scheduler.current_downsampler, GradNormDownsamplingStrategy)
    assert scheduler.next_threshold is None
