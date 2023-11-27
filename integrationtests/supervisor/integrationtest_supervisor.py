import json
import time

import yaml
from integrationtests.utils import SCRIPT_PATH, DatasetHelper, connect_to_server, get_minimal_pipeline_config
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import GetPipelineStatusRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub

POLL_TIMEOUT = 1
TIMEOUT = 15


def wait_for_pipeline(pipeline_id: int) -> str:
    req = GetPipelineStatusRequest(pipeline_id=pipeline_id)
    res = supervisor.get_pipeline_status(req)
    while res.status == "running":
        time.sleep(POLL_TIMEOUT)
        res = supervisor.get_pipeline_status(req)
    return res


def test_mnist() -> None:
    mnist_config_path = SCRIPT_PATH.parent.parent / "benchmark" / "mnist" / "mnist.yaml"
    with open(mnist_config_path, "r", encoding="utf-8") as config_file:
        pipeline_config = yaml.safe_load(config_file)
    print(pipeline_config)

    res = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
            eval_directory=".",
            start_replay_at=0,
        )
    )
    pipeline_id = res.pipeline_id

    print(f"pipeline id: {pipeline_id}")
    assert pipeline_id >= 0


def test_one_experiment_pipeline() -> None:
    pipeline_config = get_minimal_pipeline_config()

    res = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
            eval_directory=".",
            start_replay_at=0,
        )
    )
    pipeline_id = res.pipeline_id

    print(f"pipeline id: {pipeline_id}")
    assert pipeline_id >= 0

    get_pipeline_status_res = wait_for_pipeline(pipeline_id)
    print(get_pipeline_status_res)
    assert get_pipeline_status_res.status == "exit"


def test_two_experiment_pipelines() -> None:
    pipeline1_config = get_minimal_pipeline_config()
    pipeline2_config = get_minimal_pipeline_config(2)

    res1 = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline1_config)),
            eval_directory=".",
            start_replay_at=0,
            maximum_triggers=2,
        )
    )
    pipeline1_id = res1.pipeline_id

    res2 = supervisor.start_pipeline(
        StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline2_config)),
            eval_directory=".",
            start_replay_at=0,
            stop_replay_at=50,
            maximum_triggers=10,
        )
    )
    pipeline2_id = res2.pipeline_id

    print(f"pipeline ids: {pipeline1_id}, {pipeline2_id}")
    assert pipeline1_id >= 0 and pipeline2_id >= 0

    get_pipeline_status_res1 = wait_for_pipeline(pipeline1_id)
    print(get_pipeline_status_res1)
    assert get_pipeline_status_res1.status == "exit"

    get_pipeline_status_res2 = wait_for_pipeline(pipeline2_id)
    print(get_pipeline_status_res2)
    assert get_pipeline_status_res2.status == "exit"


if __name__ == "__main__":
    dataset_helper = DatasetHelper()
    try:
        dataset_helper.setup_dataset()
        supervisor_channel = connect_to_server("supervisor")
        supervisor = SupervisorStub(supervisor_channel)
        test_one_experiment_pipeline()
        test_two_experiment_pipelines()
        time.sleep(TIMEOUT)
    finally:
        dataset_helper.cleanup_dataset_dir()
        dataset_helper.cleanup_storage_database()
