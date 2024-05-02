import json
import logging
from typing import Optional

import grpc
from google.protobuf.json_format import MessageToDict
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import GetPipelineStatusRequest, GetPipelineStatusResponse
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import PipelineResponse, StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorStub
from modyn.utils import MAX_MESSAGE_SIZE, grpc_connection_established

logger = logging.getLogger(__name__)


class GRPCHandler:
    def __init__(self, client_config: dict) -> None:
        self.config = client_config
        self.connected_to_supervisor = False
        self.init_supervisor()

    def init_supervisor(self) -> None:
        assert self.config is not None
        supervisor_address = (
            f"{self.config['supervisor']['ip']}:{self.config['supervisor']['port']}"
        )
        self.supervisor_channel = grpc.insecure_channel(
            supervisor_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )

        if not grpc_connection_established(self.supervisor_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to supervisor at {supervisor_address}."
            )

        self.supervisor = SupervisorStub(self.supervisor_channel)
        logger.info("Successfully connected to supervisor.")
        self.connected_to_supervisor = True

    def start_pipeline(
        self,
        pipeline_config: dict,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> dict:
        if not self.connected_to_supervisor:
            raise ConnectionError(
                "Tried to start pipeline at supervisor, but not there is no gRPC connection."
            )

        start_pipeline_request = StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
        )

        if start_replay_at is not None:
            start_pipeline_request.start_replay_at = start_replay_at
        if stop_replay_at is not None:
            start_pipeline_request.stop_replay_at = stop_replay_at
        if maximum_triggers is not None:
            start_pipeline_request.maximum_triggers = maximum_triggers

        res: PipelineResponse = self.supervisor.start_pipeline(start_pipeline_request)
        ret = MessageToDict(
            res,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )

        return ret

    def get_pipeline_status(self, pipeline_id: int) -> dict:
        if not self.connected_to_supervisor:
            raise ConnectionError(
                "Tried to start pipeline at supervisor, but not there is no gRPC connection."
            )

        get_status_request = GetPipelineStatusRequest(pipeline_id=pipeline_id)

        res: GetPipelineStatusResponse = self.supervisor.get_pipeline_status(
            get_status_request
        )
        ret = MessageToDict(
            res,
            preserving_proto_field_name=True,
            always_print_fields_with_no_presence=True,
        )

        return ret
