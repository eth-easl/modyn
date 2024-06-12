import json
import time

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from integrationtests.utils import (
    DUMMY_CONFIG_FILE,
    TinyDatasetHelper,
    connect_to_server,
    load_config_from_file,
    ImageDatasetHelper,
    get_modyn_config,
    RHO_LOSS_CONFIG_FILE,
)
from modyn.selector.internal.grpc.generated.selector_pb2 import GetSelectionStrategyRequest
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.supervisor.internal.grpc.enums import PipelineStage, PipelineStatus
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import GetPipelineStatusRequest, GetPipelineStatusResponse
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub

POLL_INTERVAL = 1
MODYN_CONFIG = get_modyn_config()
IMAGE_DATASET_ID = "image_test_dataset"


def wait_for_pipeline(pipeline_id: int) -> GetPipelineStatusResponse:
    req = GetPipelineStatusRequest(pipeline_id=pipeline_id)
    supervisor = SupervisorStub(connect_to_server("supervisor"))
    res = supervisor.get_pipeline_status(req)
    while res.status == PipelineStatus.RUNNING:
        time.sleep(POLL_INTERVAL)
        res = supervisor.get_pipeline_status(req)

    return res


def parse_grpc_res(msg: Message) -> dict:
    return MessageToDict(msg, preserving_proto_field_name=True, always_print_fields_with_no_presence=True)


def assert_pipeline_exit_without_error(res: dict) -> None:
    assert res["status"] == PipelineStatus.EXIT

    exit_msg = {}
    for msg in res["pipeline_stage"]:
        if msg["stage"] == PipelineStage.EXIT:
            exit_msg = msg

    assert exit_msg
    assert exit_msg["exit_msg"]["exitcode"] == 0


def test_one_experiment_pipeline() -> None:
    pipeline_config = load_config_from_file(DUMMY_CONFIG_FILE)
    supervisor = SupervisorStub(connect_to_server("supervisor"))
    res = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                start_replay_at=0,
                maximum_triggers=1,
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
    supervisor = SupervisorStub(connect_to_server("supervisor"))
    res1 = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                start_replay_at=0,
                maximum_triggers=1,
            )
        )
    )
    pipeline1_id = res1["pipeline_id"]

    res2 = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                start_replay_at=0,
                stop_replay_at=100000,
                maximum_triggers=1,
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


def test_rho_loss_pipeline_with_two_triggers() -> None:
    pipeline_config = load_config_from_file(RHO_LOSS_CONFIG_FILE)
    supervisor = SupervisorStub(connect_to_server("supervisor"))
    selector = SelectorStub(connect_to_server("selector"))
    res = parse_grpc_res(
        supervisor.start_pipeline(
            StartPipelineRequest(
                pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
                start_replay_at=0,
            )
        )
    )
    pipeline_id = res["pipeline_id"]
    assert pipeline_id >= 0
    get_pipeline_status_res = parse_grpc_res(wait_for_pipeline(pipeline_id))
    assert_pipeline_exit_without_error(get_pipeline_status_res)

    # retrieve the current downsampling config
    selection_strategy_resp = parse_grpc_res(
        selector.get_selection_strategy(
            GetSelectionStrategyRequest(pipeline_id=pipeline_id)
        )
    )
    assert selection_strategy_resp["strategy_name"] == "RemoteRHOLossDownsampling"
    assert selection_strategy_resp["downsampling_enabled"]
    rho_pipeline_id = selection_strategy_resp["downsampler_config"]["rho_pipeline_id"]
    il_model_id = selection_strategy_resp["downsampler_config"]["il_model_id"]
    # validate that there are 2 triggers, 2 models in corresponding tables
    # one of the model is this il_model_id, which should be larger than the other




if __name__ == "__main__":
    tiny_dataset_helper = TinyDatasetHelper()
    try:
        tiny_dataset_helper.setup_dataset()
        test_one_experiment_pipeline()
        test_two_experiment_pipelines()
    finally:
        tiny_dataset_helper.cleanup_dataset_dir()
        tiny_dataset_helper.cleanup_storage_database()

    image_dataset_helper = ImageDatasetHelper(dataset_size=20, dataset_id=IMAGE_DATASET_ID)
    try:
        image_dataset_helper.setup_dataset()
        test_rho_loss_pipeline_with_two_triggers()
    finally:
        image_dataset_helper.cleanup_dataset_dir()
        image_dataset_helper.cleanup_storage_database()
