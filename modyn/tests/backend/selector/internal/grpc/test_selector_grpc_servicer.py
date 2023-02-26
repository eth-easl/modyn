# pylint: disable=unused-argument, no-name-in-module, redefined-outer-name

from unittest.mock import MagicMock, patch

from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (  # noqa: E402, E501, E611
    DataInformRequest,
    GetNumberOfSamplesRequest,
    GetSamplesRequest,
    JsonString,
    NumberOfSamplesResponse,
    PipelineResponse,
    RegisterPipelineRequest,
    SamplesResponse,
    TriggerResponse,
)
from modyn.backend.selector.internal.grpc.selector_grpc_servicer import SelectorGRPCServicer
from modyn.backend.selector.internal.selector_manager import SelectorManager


def noop_init_metadata_db(self):
    pass


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
def test_init():
    mgr = SelectorManager({})
    servicer = SelectorGRPCServicer(mgr)
    assert servicer.selector_manager == mgr


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "register_pipeline")
def test_register_pipeline(test_register_pipeline: MagicMock):
    mgr = SelectorManager({})
    servicer = SelectorGRPCServicer(mgr)
    request = RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value="strat"))
    test_register_pipeline.return_value = 42

    response: PipelineResponse = servicer.register_pipeline(request, None)

    assert response.pipeline_id == 42
    test_register_pipeline.assert_called_once_with(2, "strat")


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "get_sample_keys_and_weights")
def test_get_sample_keys_and_weights(test_get_sample_keys_and_weights: MagicMock):
    mgr = SelectorManager({})
    servicer = SelectorGRPCServicer(mgr)
    request = GetSamplesRequest(pipeline_id=0, trigger_id=1, worker_id=2)
    test_get_sample_keys_and_weights.return_value = [("a", 1.0), ("b", 1.0)]

    response: SamplesResponse = servicer.get_sample_keys_and_weights(request, None)

    assert response.training_samples_subset == ["a", "b"]
    assert response.training_samples_weights == [1.0, 1.0]

    test_get_sample_keys_and_weights.assert_called_once_with(0, 1, 2)


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "inform_data")
def test_inform_data(test_inform_data: MagicMock):
    mgr = SelectorManager({})
    servicer = SelectorGRPCServicer(mgr)
    request = DataInformRequest(pipeline_id=0, keys=["a", "b"], timestamps=[1, 2], labels=[0, 1])
    test_inform_data.return_value = None

    servicer.inform_data(request, None)
    test_inform_data.assert_called_once_with(0, ["a", "b"], [1, 2], [0, 1])


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "inform_data_and_trigger")
def test_inform_data_and_trigger(test_inform_data_and_trigger: MagicMock):
    mgr = SelectorManager({})
    servicer = SelectorGRPCServicer(mgr)
    request = DataInformRequest(pipeline_id=0, keys=["a", "b"], timestamps=[1, 2], labels=[0, 1])
    test_inform_data_and_trigger.return_value = 42

    response: TriggerResponse = servicer.inform_data_and_trigger(request, None)
    assert response.trigger_id == 42

    test_inform_data_and_trigger.assert_called_once_with(0, ["a", "b"], [1, 2], [0, 1])


@patch.object(SelectorManager, "init_metadata_db", noop_init_metadata_db)
@patch.object(SelectorManager, "get_number_of_samples")
def test_get_number_of_samples(test_get_number_of_samples: MagicMock):
    mgr = SelectorManager({})
    servicer = SelectorGRPCServicer(mgr)
    request = GetNumberOfSamplesRequest(pipeline_id=42, trigger_id=21)
    test_get_number_of_samples.return_value = 12

    response: NumberOfSamplesResponse = servicer.get_number_of_samples(request, None)
    assert response.num_samples == 12

    test_get_number_of_samples.assert_called_once_with(42, 21)
