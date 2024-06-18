import json
import time

from google.protobuf.json_format import MessageToDict
from google.protobuf.message import Message
from integrationtests.utils import (
    DUMMY_CONFIG_FILE,
    RHO_LOSS_CONFIG_FILE,
    TinyDatasetHelper,
    connect_to_server,
    get_modyn_config,
    load_config_from_file,
)
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Pipeline, TrainedModel, Trigger
from modyn.selector.internal.grpc.generated.selector_pb2 import GetSelectionStrategyRequest
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.supervisor.internal.grpc.enums import PipelineStage, PipelineStatus
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import GetPipelineStatusRequest, GetPipelineStatusResponse
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub

POLL_INTERVAL = 1
MODYN_CONFIG = get_modyn_config()


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
    num_triggers = 2
    # we set in the rho_loss.yaml that trigger is caused every 10 samples
    # since we have a dataset of 20 samples there will be 2 triggers.
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
        selector.get_selection_strategy(GetSelectionStrategyRequest(pipeline_id=pipeline_id))
    )
    print(selection_strategy_resp)
    assert selection_strategy_resp["strategy_name"] == "RemoteRHOLossDownsampling"
    assert selection_strategy_resp["downsampling_enabled"]
    downsampling_config = json.loads(selection_strategy_resp["downsampler_config"]["value"])
    rho_pipeline_id = downsampling_config["rho_pipeline_id"]
    latest_il_model_id = downsampling_config["il_model_id"]

    with MetadataDatabaseConnection(MODYN_CONFIG) as database:
        main_pipeline: Pipeline = database.session.query(Pipeline).filter(Pipeline.pipeline_id == pipeline_id).first()
        # validate this rho_pipeline_id is recorded in the pipelines table
        assert main_pipeline.auxiliary_pipeline_id == rho_pipeline_id
        # validate that there are {num_triggers} triggers associated with the rho_pipeline_id
        triggers: list[Trigger] = database.session.query(Trigger).filter(Trigger.pipeline_id == rho_pipeline_id).all()
        trigger_ids = [trigger.trigger_id for trigger in triggers]
        assert len(trigger_ids) == num_triggers
        # validate there are {num_triggers} models associated with the rho_pipeline_id.
        models: list[TrainedModel] = (
            database.session.query(TrainedModel).filter(TrainedModel.pipeline_id == rho_pipeline_id).all()
        )
        assert len(models) == num_triggers
        assert all(model.trigger_id in trigger_ids for model in models)
        model_ids = [model.model_id for model in models]
        assert latest_il_model_id == max(model_ids)

        # validate the holdout set is in correct proportion
        # in rho_loss.yaml, each trigger contains 10 images. The holdout set ratio is 30%
        # so the holdout set should contain 3 images
        assert all(trigger.num_keys == 3 for trigger in triggers)


if __name__ == "__main__":
    tiny_dataset_helper = TinyDatasetHelper(dataset_size=20)
    try:
        tiny_dataset_helper.setup_dataset()
        test_two_experiment_pipelines()
        test_rho_loss_pipeline_with_two_triggers()
    finally:
        tiny_dataset_helper.cleanup_dataset_dir()
        tiny_dataset_helper.cleanup_storage_database()
