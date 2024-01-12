import json
import time

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from integrationtests.utils import (
    DUMMY_CONFIG_FILE,
    MNIST_CONFIG_FILE,
    TinyDatasetHelper,
    connect_to_server,
    load_config_from_file,
)
from modyn.supervisor.internal.grpc.enums import PipelineStage, PipelineStatus
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import GetPipelineStatusRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub

POLL_TIMEOUT = 1


def wait_for_pipeline(pipeline_id: int) -> str:
    req = GetPipelineStatusRequest(pipeline_id=pipeline_id)

    res = supervisor.get_pipeline_status(req)
    while res.status == PipelineStatus.RUNNING:
        time.sleep(POLL_TIMEOUT)
        res = supervisor.get_pipeline_status(req)

    return res


def parse_grpc_res(msg: Message) -> dict:
    return MessageToDict(msg, preserving_proto_field_name=True, including_default_value_fields=True)


def assert_pipeline_exit_without_error(res: dict) -> None:
    assert res["status"] == PipelineStatus.EXIT

    exit_msg = {}
    for msg in res["pipeline_stage"]:
        if msg["stage"] == PipelineStage.EXIT:
            exit_msg = msg

    assert exit_msg
    assert exit_msg["exit_msg"]["exitcode"] == 0


def test_mnist() -> None:
    pipeline_config = load_config_from_file(MNIST_CONFIG_FILE)
    print(pipeline_config)

    res = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                eval_directory=".",
                start_replay_at=0,
                maximum_triggers=1,
                evaluation_matrix=False,
            )
        )
    )
    pipeline_id = res["pipeline_id"]

    print(f"start pipeline: {res}")
    assert pipeline_id >= 0

    get_pipeline_status_res = parse_grpc_res(wait_for_pipeline(pipeline_id))
    print(get_pipeline_status_res)
    assert_pipeline_exit_without_error(get_pipeline_status_res)


def test_one_experiment_pipeline() -> None:
    pipeline_config = load_config_from_file(DUMMY_CONFIG_FILE)

    res = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                eval_directory=".",
                start_replay_at=0,
                maximum_triggers=1,
                evaluation_matrix=True,
            )
        )
    )
    pipeline_id = res["pipeline_id"]

    print(f"start pipeline: {res}")
    assert pipeline_id >= 0

    get_pipeline_status_res = parse_grpc_res(wait_for_pipeline(pipeline_id))
    print(get_pipeline_status_res)
    assert_pipeline_exit_without_error(get_pipeline_status_res)


def test_two_experiment_pipelines() -> None:
    pipeline_config = load_config_from_file(DUMMY_CONFIG_FILE)

    res1 = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                eval_directory=".",
                start_replay_at=0,
                maximum_triggers=1,
                evaluation_matrix=False,
            )
        )
    )
    pipeline1_id = res1["pipeline_id"]

    res2 = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                eval_directory=".",
                start_replay_at=0,
                stop_replay_at=100000,
                maximum_triggers=1,
                evaluation_matrix=False,
            )
        )
    )
    pipeline2_id = res2["pipeline_id"]

    print(f"start pipeline1: {res1}")
    print(f"start pipeline2: {res2}")
    assert pipeline1_id >= 0 and pipeline2_id >= 0

    get_pipeline_status_res1 = parse_grpc_res(wait_for_pipeline(pipeline1_id))
    print(pipeline1_id, get_pipeline_status_res1)
    get_pipeline_status_res2 = parse_grpc_res(wait_for_pipeline(pipeline2_id))
    print(pipeline2_id, get_pipeline_status_res2)

    assert_pipeline_exit_without_error(get_pipeline_status_res1)
    assert_pipeline_exit_without_error(get_pipeline_status_res2)


if __name__ == "__main__":
    dataset_helper = TinyDatasetHelper()
    try:
        dataset_helper.setup_dataset()
        supervisor_channel = connect_to_server("supervisor")
        supervisor = SupervisorStub(supervisor_channel)
        test_one_experiment_pipeline()
        test_two_experiment_pipelines()
    finally:
        dataset_helper.cleanup_dataset_dir()
        dataset_helper.cleanup_storage_database()
