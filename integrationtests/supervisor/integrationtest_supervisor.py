import json

from integrationtests.utils import DatasetHelper, connect_to_server, get_minimal_pipeline_config
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub


def test_start_one_pipeline() -> None:
    supervisor_channel = connect_to_server("supervisor")
    supervisor = SupervisorStub(supervisor_channel)

    pipeline_config = get_minimal_pipeline_config()

    pipeline_id = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
            eval_directory=".",
        )
    ).pipeline_id

    print(f"pipeline id: {pipeline_id}")
    assert pipeline_id >= 0


def test_start_two_pipelines() -> None:
    supervisor_channel = connect_to_server("supervisor")
    supervisor = SupervisorStub(supervisor_channel)

    pipeline1_config = get_minimal_pipeline_config()
    pipeline2_config = get_minimal_pipeline_config(2)

    pipeline1_id = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline1_config)),
            eval_directory=".",
        )
    ).pipeline_id

    pipeline2_id = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline2_config)),
            eval_directory=".",
            start_replay_at=0,
            stop_replay_at=50,
            maximum_triggers=10,
        )
    ).pipeline_id

    print(f"pipeline ids: {pipeline1_id}, {pipeline2_id}")
    assert pipeline1_id >= 0 and pipeline2_id >= 0


if __name__ == "__main__":
    dataset_helper = DatasetHelper()
    try:
        dataset_helper.setup_dataset()
        test_start_one_pipeline()
        test_start_two_pipelines()
    finally:
        dataset_helper.cleanup_dataset_dir()
        dataset_helper.cleanup_storage_database()
