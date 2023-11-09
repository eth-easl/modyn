import json
import time

import yaml
from integrationtests.utils import SCRIPT_PATH, DatasetHelper, connect_to_server, get_minimal_pipeline_config
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub

TIMEOUT = 60


def test_mnist() -> None:
    supervisor_channel = connect_to_server("supervisor")
    supervisor = SupervisorStub(supervisor_channel)

    mnist_config_path = SCRIPT_PATH.parent.parent / "benchmark" / "mnist" / "mnist.yaml"
    with open(mnist_config_path, "r", encoding="utf-8") as config_file:
        pipeline_config = yaml.safe_load(config_file)
    print(pipeline_config)

    pipeline_id = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
            eval_directory=".",
            start_replay_at=0,
            maximum_triggers=3,
        )
    ).pipeline_id

    print(f"pipeline id: {pipeline_id}")
    assert pipeline_id >= 0


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
        time.sleep(TIMEOUT)
        # test_mnist()
        # time.sleep(TIMEOUT * 5)
        # TODO(#317): implement get_status proto. check for pipeline execution success.
    finally:
        dataset_helper.cleanup_dataset_dir()
        dataset_helper.cleanup_storage_database()
