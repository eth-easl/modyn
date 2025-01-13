import json
import logging
import os
import threading

# pylint: disable=no-name-in-module
import grpc
from google.protobuf.json_format import ParseDict

from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import (
    GetPipelineStatusRequest,
    GetPipelineStatusResponse,
    PipelineResponse,
    StartPipelineRequest,
)
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorServicer  # noqa: E402, E501
from modyn.supervisor.internal.supervisor import Supervisor

logger = logging.getLogger(__name__)


class SupervisorGRPCServicer(SupervisorServicer):
    """Provides methods that implement functionality of the selector."""

    def __init__(self, supervisor: Supervisor, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self._supervisor = supervisor
        self._supervisor.init_cluster_connection()

    def start_pipeline(self, request: StartPipelineRequest, context: grpc.ServicerContext) -> PipelineResponse:
        tid = threading.get_native_id()
        pid = os.getpid()
        logger.info(f"[{pid}][{tid}]: Starting pipeline with request - {request}")

        start_replay_at: int | None = None
        if request.HasField("start_replay_at"):
            start_replay_at = request.start_replay_at
        stop_replay_at: int | None = None
        if request.HasField("stop_replay_at"):
            stop_replay_at = request.stop_replay_at
        maximum_triggers: int | None = None
        if request.HasField("maximum_triggers"):
            maximum_triggers = request.maximum_triggers

        pipeline_config = json.loads(request.pipeline_config.value)
        msg = self._supervisor.start_pipeline(
            pipeline_config,
            self.modyn_config["supervisor"]["eval_directory"],
            start_replay_at,
            stop_replay_at,
            maximum_triggers,
        )
        return ParseDict(msg, PipelineResponse())

    def get_pipeline_status(
        self, request: GetPipelineStatusRequest, context: grpc.ServicerContext
    ) -> GetPipelineStatusResponse:
        msg = self._supervisor.get_pipeline_status(request.pipeline_id)
        return ParseDict(msg, GetPipelineStatusResponse())
