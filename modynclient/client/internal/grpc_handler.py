import json
import logging
import pathlib
from typing import Optional

import grpc

# TODO(#317 client): share with modyn or make a copy?
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import (
    StartPipelineRequest, 
    GetPipelineStatusRequest, 
    GetPipelineStatusResponse,
)
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
        supervisor_address = f"{self.config['supervisor']['ip']}:{self.config['supervisor']['port']}"
        self.supervisor_channel = grpc.insecure_channel(
            supervisor_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )

        if not grpc_connection_established(self.supervisor_channel):
            raise ConnectionError(f"Could not establish gRPC connection to supervisor at {supervisor_address}.")

        self.supervisor = SupervisorStub(self.supervisor_channel)
        logger.info("Successfully connected to supervisor.")
        self.connected_to_supervisor = True
    
    def start_pipeline(
        self,
        pipeline_config: dict,
        eval_directory: pathlib.Path,
        start_replay_at: Optional[int] = None,
        stop_replay_at: Optional[int] = None,
        maximum_triggers: Optional[int] = None,
    ) -> int:
        if not self.connected_to_supervisor:
            raise ConnectionError("Tried to start pipeline at supervisor, but not there is no gRPC connection.")
        
        start_pipeline_request = StartPipelineRequest(
            pipeline_config=SupervisorJsonString(value=json.dumps(pipeline_config)),
            eval_directory=str(eval_directory),
        )

        if start_replay_at is not None:
            start_pipeline_request.start_replay_at = start_replay_at
        if stop_replay_at is not None:
            start_pipeline_request.stop_replay_at = stop_replay_at
        if maximum_triggers is not None:
            start_pipeline_request.maximum_triggers = maximum_triggers

        pipeline_id = self.supervisor.start_pipeline(start_pipeline_request).pipeline_id

        return pipeline_id
    
    def get_pipeline_status(self, pipeline_id: int) -> dict:
        if not self.connected_to_supervisor:
            raise ConnectionError("Tried to start pipeline at supervisor, but not there is no gRPC connection.")
        
        get_status_request = GetPipelineStatusRequest(pipeline_id=pipeline_id)

        res: GetPipelineStatusResponse = self.supervisor.get_pipeline_status(get_status_request)
        status = res.status
        detail = json.loads(res.detail.value)

        return {"status": status, "detail": detail}