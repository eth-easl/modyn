

import grpc
from integrationtests.utils import get_modyn_config
from modyn.utils import grpc_connection_established
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub

from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (
    RegisterPipelineRequest,
    DataInformRequest,
    GetSamplesRequest,
)


def connect_to_selector_servicer() -> grpc.Channel:
    config = get_modyn_config()

    selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"
    selector_channel = grpc.insecure_channel(selector_address)

    if not grpc_connection_established(selector_channel):
        assert False, f"Could not establish gRPC connection to selector at {selector_address}."

    return selector_channel


def test_selector() -> None:
    # Register a finetuning pipeline (which I'm assuming the initial training will be)
    selector_channel = connect_to_selector_servicer()
    selector = SelectorStub(selector_channel)

    strategy_config = {
        "name": "finetune", "configs": {"limit": 8, "reset_after_trigger": True}
    }

    pipeline_id = selector.register_pipeline(RegisterPipelineRequest(
        num_workers=2, selector_strategy_config=strategy_config)).pipeline_id

    selector.inform_data(DataInformRequest(
        pipeline_id=pipeline_id,
        keys=['key_0', 'key_1', 'key_2'],
        timestamps=[1, 2, 3],
        labels=[1, 0, 1],
    ))

    trigger_id = selector.inform_data_and_trigger(DataInformRequest(
        pipeline_id=pipeline_id,
        keys=['key_3', 'key_4', 'key_5'],
        timestamps=[4, 5, 6],
        labels=[0, 0, 1],
    )).trigger_id

    worker_1_samples = selector.get_sample_keys(
        GetSamplesRequest(
            pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0
        )
    ).training_samples_subset

    worker_2_samples = selector.get_sample_keys(
        GetSamplesRequest(
            pipeline_id=pipeline_id, trigger_id=trigger_id, worker_id=0
        )
    ).training_samples_subset

    assert set(worker_1_samples + worker_2_samples) == set(['key_' + str(i) for i in range(6)])


if __name__ == '__main__':
    test_selector()
