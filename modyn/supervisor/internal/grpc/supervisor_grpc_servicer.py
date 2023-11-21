import json
import logging
import os
import threading

# pylint: disable=no-name-in-module
from typing import Optional

import grpc
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import (
    StartPipelineRequest,
    PipelineResponse,
    GetPipelineStatusRequest,
    GetPipelineStatusResponse,
)
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import JsonString as SupervisorJsonString
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorServicer  # noqa: E402, E501
from modyn.supervisor.internal.supervisor import Supervisor

logger = logging.getLogger(__name__)


class SupervisorGRPCServicer(SupervisorServicer):
    """Provides methods that implement functionality of the selector."""

    def __init__(self, supervisor: Supervisor, modyn_config: dict) -> None:
        self.modyn_config = modyn_config
        self._supervisor = supervisor
        self._supervisor.init_cluster_connection()
        self._supervisor.monitor_pipelines()

    def start_pipeline(self, request: StartPipelineRequest, context: grpc.ServicerContext) -> PipelineResponse:
        tid = threading.get_native_id()
        pid = os.getpid()

        logger.info(f"[{pid}][{tid}]: Starting pipeline with request - {request}")

        start_replay_at: Optional[int] = None
        if request.HasField("start_replay_at"):
            start_replay_at = request.start_replay_at
        stop_replay_at: Optional[int] = None
        if request.HasField("stop_replay_at"):
            stop_replay_at = request.stop_replay_at
        maximum_triggers: Optional[int] = None
        if request.HasField("maximum_triggers"):
            maximum_triggers = request.maximum_triggers

        pipeline_config = json.loads(request.pipeline_config.value)
        logger.info(f"[{pid}][{tid}]: pipeline config {pipeline_config}")
        pipeline_id = self._supervisor.start_pipeline(
            pipeline_config,
            request.eval_directory,
            start_replay_at,
            stop_replay_at,
            maximum_triggers,
        )

        return PipelineResponse(pipeline_id=pipeline_id)

    def get_pipeline_status(
        self, 
        request: GetPipelineStatusRequest, 
        context: grpc.ServicerContext
    ) -> GetPipelineStatusResponse:
        res = self._supervisor.get_pipeline_status(request.pipeline_id)

        return GetPipelineStatusResponse(
            status = res["status"],
            detail = SupervisorJsonString(value=json.dumps(res["detail"]))
        )
