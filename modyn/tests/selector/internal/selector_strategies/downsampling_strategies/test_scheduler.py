from unittest.mock import patch

import pytest
from modyn.config import CoresetStrategyConfig, MultiDownsamplingConfig
from modyn.config.schema.pipeline.sampling.downsampling_config import (
    GradNormDownsamplingConfig,
    LossDownsamplingConfig,
    NoDownsamplingConfig,
)
from modyn.selector.internal.selector_strategies.downsampling_strategies import (
    AbstractDownsamplingStrategy,
    DownsamplingScheduler,
    GradNormDownsamplingStrategy,
    LossDownsamplingStrategy,
    instantiate_scheduler,
)
from modyn.tests.selector.internal.storage_backend.utils import MockStorageBackend
from modyn.utils import DownsamplingMode


def get_configs():
    return [
        LossDownsamplingConfig(ratio=50, sample_then_batch=True),
        GradNormDownsamplingConfig(ratio=25, sample_then_batch=False),
    ]


def get_configs_triple():
    return [
        LossDownsamplingConfig(ratio=50, sample_then_batch=True),
        GradNormDownsamplingConfig(ratio=25, sample_then_batch=False),
        NoDownsamplingConfig(),
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
    pipeline_id = 0
    maximum_keys_in_memory = 1000
    downs = DownsamplingScheduler({}, pipeline_id, conf, [12], maximum_keys_in_memory)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, LossDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH
    assert downs.current_downsampler.downsampling_ratio == 50
    assert downs.next_threshold == 12

    selector_storage_backend = MockStorageBackend(pipeline_id, {}, maximum_keys_in_memory)
    downs.inform_next_trigger(12, selector_storage_backend)

    assert downs.maximum_keys_in_memory == 1000
    assert isinstance(downs.current_downsampler, GradNormDownsamplingStrategy)
    assert downs.current_downsampler.requires_remote_computation
    assert downs.current_downsampler.downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE
    assert downs.current_downsampler.downsampling_ratio == 25
    assert downs.next_threshold is None
    mock_inform_next_trigger.assert_called_once_with(12, selector_storage_backend)


def test_switch_functions():
    conf = get_configs()
    pipeline_id = 0
    maximum_keys_in_memory = 1000
    downs = DownsamplingScheduler({}, pipeline_id, conf, [12], maximum_keys_in_memory)
    selector_storage_backend = MockStorageBackend(pipeline_id, {}, maximum_keys_in_memory)
    # below the threshold
    for i in range(12):
        downs.inform_next_trigger(i, selector_storage_backend)
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
        downs.inform_next_trigger(i, selector_storage_backend)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.downsampling_strategy == "RemoteGradNormDownsampling"
        assert downs.training_status_bar_scale == 25


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
    pipeline_id = 0
    maximum_keys_in_memory = 1000
    downs = DownsamplingScheduler({}, pipeline_id, conf, [12, 15], maximum_keys_in_memory)
    selector_storage_backend = MockStorageBackend(pipeline_id, {}, maximum_keys_in_memory)
    # below the first threshold
    for i in range(12):
        downs.inform_next_trigger(i, selector_storage_backend)
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
        downs.inform_next_trigger(i, selector_storage_backend)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.downsampling_strategy == "RemoteGradNormDownsampling"
        assert downs.training_status_bar_scale == 25

    # above the last threshold
    for i in range(15, 25):
        downs.inform_next_trigger(i, selector_storage_backend)
        assert not downs.requires_remote_computation
        assert downs.downsampling_params == {}
        assert downs.downsampling_strategy == ""
        assert downs.training_status_bar_scale == 100


def test_wrong_trigger():
    conf = get_configs()
    pipeline_id = 0
    maximum_keys_in_memory = 1000
    downs = DownsamplingScheduler({}, pipeline_id, conf, [12], maximum_keys_in_memory)
    selector_storage_backend = MockStorageBackend(pipeline_id, {}, maximum_keys_in_memory)
    for i in range(12):
        downs.inform_next_trigger(i, selector_storage_backend)
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
    downs.inform_next_trigger(13, selector_storage_backend)
    assert downs.requires_remote_computation

    # then ask again for triggers below the threshold, you still get the second downsampler
    for i in range(12):
        downs.inform_next_trigger(i, selector_storage_backend)
        assert downs.requires_remote_computation
        assert downs.downsampling_params == {
            "downsampling_ratio": 25,
            "maximum_keys_in_memory": 1000,
            "sample_then_batch": False,
        }
        assert downs.downsampling_strategy == "RemoteGradNormDownsampling"
        assert downs.training_status_bar_scale == 25


def test_instantiate_scheduler_just_one():
    config = CoresetStrategyConfig(
        downsampling_config=LossDownsamplingConfig(ratio=50, sample_then_batch=True),
        maximum_keys_in_memory=1000,
        tail_triggers=None,
    )
    scheduler = instantiate_scheduler(config, {}, 0)
    assert isinstance(scheduler.current_downsampler, LossDownsamplingStrategy)
    assert scheduler.next_threshold is None


def test_instantiate_scheduler_list():
    maximum_keys_in_memory = 123
    config = CoresetStrategyConfig(
        downsampling_config=MultiDownsamplingConfig(downsampling_list=get_configs(), downsampling_thresholds=[7]),
        maximum_keys_in_memory=maximum_keys_in_memory,
        tail_triggers=None,
    )
    pipeline_id = 0

    selector_storage_backend = MockStorageBackend(pipeline_id, {}, maximum_keys_in_memory)
    scheduler = instantiate_scheduler(config, {}, 0)

    assert isinstance(scheduler.current_downsampler, LossDownsamplingStrategy)
    assert scheduler.next_threshold == 7

    scheduler.inform_next_trigger(8, selector_storage_backend)
    assert scheduler.requires_remote_computation

    assert isinstance(scheduler.current_downsampler, GradNormDownsamplingStrategy)
    assert scheduler.next_threshold is None
