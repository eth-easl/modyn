# pylint: disable=unused-argument, no-name-in-module, redefined-outer-name

import tempfile
from typing import Iterable
from unittest.mock import MagicMock, patch

from modyn.selector.internal.grpc.generated.selector_pb2 import (  # noqa: E402, E501, E611
    DataInformRequest,
    GetNumberOfPartitionsRequest,
    GetNumberOfSamplesRequest,
    GetSamplesRequest,
    GetSelectionStrategyRequest,
    NumberOfPartitionsResponse,
    NumberOfSamplesResponse,
    SamplesResponse,
    SelectionStrategyResponse,
    TriggerResponse,
)
from modyn.selector.internal.grpc.selector_grpc_servicer import SelectorGRPCServicer
from modyn.selector.internal.selector_manager import SelectorManager


def get_minimal_modyn_config():
    return {"selector": {"keys_in_selector_cache": 1000, "trigger_sample_directory": "/does/not/exist"}}


def test_init():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        assert servicer.selector_manager == mgr


@patch.object(SelectorManager, "get_sample_keys_and_weights")
def test_get_sample_keys_and_weights(test_get_sample_keys_and_weights: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        request = GetSamplesRequest(pipeline_id=0, trigger_id=1, worker_id=2, partition_id=3)
        test_get_sample_keys_and_weights.return_value = [(10, 1.0), (11, 1.0)]

        responses: Iterable[SamplesResponse] = list(servicer.get_sample_keys_and_weights(request, None))
        assert len(responses) == 1
        response = responses[0]
        assert response.training_samples_subset == [10, 11]
        assert response.training_samples_weights == [1.0, 1.0]

        test_get_sample_keys_and_weights.assert_called_once_with(0, 1, 2, 3)


@patch.object(SelectorManager, "get_sample_keys_and_weights")
def test_get_sample_keys_and_weights_batching(test_get_sample_keys_and_weights: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)

        servicer = SelectorGRPCServicer(mgr, 1)
        request = GetSamplesRequest(pipeline_id=0, trigger_id=1, worker_id=2, partition_id=3)
        test_get_sample_keys_and_weights.return_value = [(10, 1.0), (11, 1.0)]

        responses: Iterable[SamplesResponse] = list(servicer.get_sample_keys_and_weights(request, None))
        assert len(responses) == 2
        response1 = responses[0]
        assert response1.training_samples_subset == [10]
        assert response1.training_samples_weights == [1.0]

        response2 = responses[1]
        assert response2.training_samples_subset == [11]
        assert response2.training_samples_weights == [1.0]

        test_get_sample_keys_and_weights.assert_called_once_with(0, 1, 2, 3)


@patch.object(SelectorManager, "get_sample_keys_and_weights")
def test_get_sample_keys_and_weights_empty(test_get_sample_keys_and_weights: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        request = GetSamplesRequest(pipeline_id=0, trigger_id=1, worker_id=2, partition_id=3)
        test_get_sample_keys_and_weights.return_value = []

        responses: Iterable[SamplesResponse] = list(servicer.get_sample_keys_and_weights(request, None))
        assert len(responses) == 1
        response1 = responses[0]
        assert response1.training_samples_subset == []
        assert response1.training_samples_weights == []

        test_get_sample_keys_and_weights.assert_called_once_with(0, 1, 2, 3)


@patch.object(SelectorManager, "inform_data")
def test_inform_data(test_inform_data: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        request = DataInformRequest(pipeline_id=0, keys=[10, 11], timestamps=[1, 2], labels=[0, 1])
        test_inform_data.return_value = None

        servicer.inform_data(request, None)
        test_inform_data.assert_called_once_with(0, [10, 11], [1, 2], [0, 1])


@patch.object(SelectorManager, "inform_data_and_trigger")
def test_inform_data_and_trigger(test_inform_data_and_trigger: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        request = DataInformRequest(pipeline_id=0, keys=[10, 11], timestamps=[1, 2], labels=[0, 1])
        test_inform_data_and_trigger.return_value = (42, {})

        response: TriggerResponse = servicer.inform_data_and_trigger(request, None)
        assert response.trigger_id == 42

        test_inform_data_and_trigger.assert_called_once_with(0, [10, 11], [1, 2], [0, 1])


@patch.object(SelectorManager, "get_number_of_samples")
def test_get_number_of_samples(test_get_number_of_samples: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        request = GetNumberOfSamplesRequest(pipeline_id=42, trigger_id=21)
        test_get_number_of_samples.return_value = 12

        response: NumberOfSamplesResponse = servicer.get_number_of_samples(request, None)
        assert response.num_samples == 12

        test_get_number_of_samples.assert_called_once_with(42, 21)


@patch.object(SelectorManager, "get_number_of_partitions")
def test__get_number_of_partitions(test_get_number_of_partitions: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        request = GetNumberOfPartitionsRequest(pipeline_id=42, trigger_id=21)
        test_get_number_of_partitions.return_value = 12

        response: NumberOfPartitionsResponse = servicer.get_number_of_partitions(request, None)
        assert response.num_partitions == 12

        test_get_number_of_partitions.assert_called_once_with(42, 21)


@patch.object(SelectorGRPCServicer, "get_selection_strategy")
def test_get_selection_strategy(test_get_selection_strategy_remote: MagicMock):
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = get_minimal_modyn_config()
        config["selector"]["trigger_sample_directory"] = tmp_dir
        mgr = SelectorManager(config)
        servicer = SelectorGRPCServicer(mgr, 8096)
        request = GetSelectionStrategyRequest(pipeline_id=42)
        test_get_selection_strategy_remote.return_value = False, "", {}

        response: SelectionStrategyResponse = servicer.get_selection_strategy(request, None)
        assert response == (False, "", {})
