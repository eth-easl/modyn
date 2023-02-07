import json

import grpc
from integrationtests.utils import get_modyn_config
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (
    DataInformRequest,
    GetSamplesRequest,
    JsonString,
    RegisterPipelineRequest,
    SamplesResponse,
)
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.utils import grpc_connection_established

# TODO(54): Write more integration tests for different strategies.


def connect_to_selector_servicer() -> grpc.Channel:
    config = get_modyn_config()

    selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"
    selector_channel = grpc.insecure_channel(selector_address)

    if not grpc_connection_established(selector_channel):
        raise ConnectionError(f"Could not establish gRPC connection to selector at {selector_address}.")

    return selector_channel


def test_newdata() -> None:
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)
    # We test the NewData strategy for finetuning on the new data, i.e., we reset without limit

    strategy_config = {"name": "NewDataStrategy", "config": {"limit": -1, "reset_after_trigger": True}}

    pipeline_id = selector.register_pipeline(
        RegisterPipelineRequest(num_workers=2, selection_strategy=JsonString(value=json.dumps(strategy_config)))
    ).pipeline_id

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=["key_0", "key_1", "key_2"],
            timestamps=[1, 2, 3],
            labels=[1, 0, 1],
        )
    )

    trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=["key_3", "key_4", "key_5"],
            timestamps=[4, 5, 6],
            labels=[0, 0, 1],
        )
    ).trigger_id

    worker1_response: SamplesResponse = selector.get_sample_keys_and_weights(
        GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0)
    )

    worker2_response: SamplesResponse = selector.get_sample_keys_and_weights(
        GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=1)
    )

    worker_1_samples = list(worker1_response.training_samples_subset)
    worker_2_samples = list(worker2_response.training_samples_subset)

    assert set(worker_1_samples + worker_2_samples) == set(
        ["key_" + str(i) for i in range(6)]
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(worker_1_samples) == 3
    assert len(worker_2_samples) == 3

    selector.inform_data(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=["key_6", "key_7", "key_8"],
            timestamps=[7, 8, 9],
            labels=[1, 0, 1],
        )
    )

    next_trigger_id = selector.inform_data_and_trigger(
        DataInformRequest(
            pipeline_id=pipeline_id,
            keys=["key_9", "key_10", "key_11"],
            timestamps=[10, 11, 12],
            labels=[0, 0, 1],
        )
    ).trigger_id

    assert next_trigger_id > trigger_id

    worker1_response: SamplesResponse = selector.get_sample_keys_and_weights(
        GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=next_trigger_id, worker_id=0)
    )

    worker2_response: SamplesResponse = selector.get_sample_keys_and_weights(
        GetSamplesRequest(pipeline_id=pipeline_id, trigger_id=next_trigger_id, worker_id=1)
    )

    worker_1_samples = list(worker1_response.training_samples_subset)
    worker_2_samples = list(worker2_response.training_samples_subset)

    assert set(worker_1_samples + worker_2_samples) == set(
        ["key_" + str(i) for i in range(6, 12)]
    ), f"got worker1 samples= {worker_1_samples}, worker2 samples={worker_2_samples}"
    assert len(worker_1_samples) == 3
    assert len(worker_2_samples) == 3


if __name__ == "__main__":
    test_newdata()
