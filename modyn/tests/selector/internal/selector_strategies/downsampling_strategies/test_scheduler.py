import pytest
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    DownsamplingScheduler,
    GradNormDownsamplingStrategy,
    LossDownsamplingStrategy,
    NoDownsamplingStrategy,
    instantiate_scheduler,
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
    downs = DownsamplingScheduler(conf, [12], 1000)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, LossDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert downs.current_downsampler.downsampling_ratio == 50
    assert downs.next_threshold == 12


def test_switch_downsamplers():
    conf = get_configs()
    downs = DownsamplingScheduler(conf, [12], 1000)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, LossDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert downs.current_downsampler.downsampling_ratio == 50
    assert downs.next_threshold == 12

    downs.get_requires_remote_computation(12)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, GradNormDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE
    assert downs.current_downsampler.downsampling_ratio == 25
    assert downs.next_threshold is None


def test_switch_functions():
    conf = get_configs()
    downs = DownsamplingScheduler(conf, [12], 1000)

    # below the threshold
    for i in range(12):
        assert downs.get_requires_remote_computation(i)
        assert downs.get_downsampling_params(i) == {
            "downsampling_period": 1,
            "downsampling_ratio": 50,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": True,
        }
        assert downs.get_downsampling_strategy(i) == "RemoteLossDownsampling"
        assert downs.get_training_status_bar_scale(i) == 50

    # above the threshold
    for i in range(12, 20):
        assert downs.get_requires_remote_computation(i)
        assert downs.get_downsampling_params(i) == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.get_downsampling_strategy(i) == "RemoteGradNormDownsampling"
        assert downs.get_training_status_bar_scale(i) == 100


def test_wrong_number_threshold():
    conf = get_configs_triple()
    # just one threshold
    with pytest.raises(ValueError):
        DownsamplingScheduler(conf, [12], 560)

    # three thresholds
    with pytest.raises(ValueError):
        DownsamplingScheduler(conf, [12, 13, 14], 560)

    # not sorted
    with pytest.raises(ValueError):
        DownsamplingScheduler(conf, [12, 11], 560)

    # double threshold
    with pytest.raises(ValueError):
        DownsamplingScheduler(conf, [12, 12], 560)

    # valid one
    DownsamplingScheduler(conf, [12, 15], 560)


def test_double_threshold():
    conf = get_configs_triple()
    downs = DownsamplingScheduler(conf, [12, 15], 1000)

    # below the first threshold
    for i in range(12):
        assert downs.get_requires_remote_computation(i)
        assert downs.get_downsampling_params(i) == {
            "downsampling_period": 1,
            "downsampling_ratio": 50,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": True,
        }
        assert downs.get_downsampling_strategy(i) == "RemoteLossDownsampling"
        assert downs.get_training_status_bar_scale(i) == 50

    # above the first threshold, below the second one
    for i in range(12, 15):
        assert downs.get_requires_remote_computation(i)
        assert downs.get_downsampling_params(i) == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.get_downsampling_strategy(i) == "RemoteGradNormDownsampling"
        assert downs.get_training_status_bar_scale(i) == 100

    # above the last threshold
    for i in range(15, 25):
        assert not downs.get_requires_remote_computation(i)
        assert downs.get_downsampling_params(i) == {}
        assert downs.get_downsampling_strategy(i) == ""
        assert downs.get_training_status_bar_scale(i) == 100


def test_wrong_trigger():
    conf = get_configs()
    downs = DownsamplingScheduler(conf, [12], 1000)

    for i in range(12):
        assert downs.get_requires_remote_computation(i)
        assert downs.get_downsampling_params(i) == {
            "downsampling_period": 1,
            "downsampling_ratio": 50,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": True,
        }
        assert downs.get_downsampling_strategy(i) == "RemoteLossDownsampling"
        assert downs.get_training_status_bar_scale(i) == 50

    # pass the threshold for the first time
    downs.get_requires_remote_computation(13)

    # then ask again for triggers below the threshold, you still get the second downsampler
    for i in range(12):
        assert downs.get_requires_remote_computation(i)
        assert downs.get_downsampling_params(i) == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.get_downsampling_strategy(i) == "RemoteGradNormDownsampling"
        assert downs.get_training_status_bar_scale(i) == 100


def test_instantiate_scheduler_empty():
    scheduler = instantiate_scheduler({}, 100)
    assert isinstance(scheduler.current_downsampler, NoDownsamplingStrategy)
    assert scheduler.next_threshold is None


def test_instantiate_scheduler_just_one():
    config = {"downsampling_config": {"strategy": "Loss", "sample_then_batch": True, "ratio": 50}}
    scheduler = instantiate_scheduler(config, 100)
    assert isinstance(scheduler.current_downsampler, LossDownsamplingStrategy)
    assert scheduler.next_threshold is None


def test_instantiate_scheduler_list():
    config = {"downsampling_config": {"downsampling_list": get_configs(), "downsampling_thresholds": 7}}
    scheduler = instantiate_scheduler(config, 123)

    assert isinstance(scheduler.current_downsampler, LossDownsamplingStrategy)
    assert scheduler.next_threshold == 7

    scheduler.get_requires_remote_computation(8)

    assert isinstance(scheduler.current_downsampler, GradNormDownsamplingStrategy)
    assert scheduler.next_threshold is None
