import grpc
from integrationtests.utils import get_minimal_pipeline_config, get_modyn_config
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub
from modyn.utils import grpc_connection_established


def connect_to_supervisor_servicer() -> grpc.Channel:
    config = get_modyn_config()

    supervisor_address = f"{config['supervisor']['hostname']}:{config['supervisor']['port']}"
    supervisor_channel = grpc.insecure_channel(supervisor_address)

    if not grpc_connection_established(supervisor_channel):
        raise ConnectionError(f"Could not establish gRPC connection to supervisor at {supervisor_address}.")

    return supervisor_channel


def test_start_one_pipeline() -> None:
    supervisor_channel = connect_to_supervisor_servicer()
    supervisor = SupervisorStub(supervisor_channel)

    pipeline_config = get_minimal_pipeline_config()

    pipeline_id = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=pipeline_config,
            eval_directory=".",
        )
    ).pipeline_id

    assert pipeline_id == 0


def test_start_two_pipelines() -> None:
    supervisor_channel = connect_to_supervisor_servicer()
    supervisor = SupervisorStub(supervisor_channel)

    pipeline1_config = get_minimal_pipeline_config()
    pipeline2_config = get_minimal_pipeline_config(2)

    pipeline1_id = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=pipeline1_config,
            eval_directory=".",
        )
    ).pipeline_id

    pipeline2_id = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=pipeline2_config,
            eval_directory=".",
            start_replay_at=0,
            stop_replay_at=50,
            maximum_triggers=10,
        )
    ).pipeline_id

    assert pipeline1_id == 0 and pipeline2_id == 1


if __name__ == "__main__":
    test_start_one_pipeline()
    test_start_two_pipelines()