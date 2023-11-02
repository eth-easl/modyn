import json
import logging
import os
import threading

import grpc

# pylint: disable=no-name-in-module
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import PipelineResponse, StartPipelineRequest
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorServicer  # noqa: E402, E501
from modyn.supervisor.supervisor import Supervisor

logger = logging.getLogger(__name__)


class SupervisorGRPCServicer(SupervisorServicer):
    """Provides methods that implement functionality of the selector."""

    def __init__(self, supervisor: Supervisor) -> None:
        self._supervisor = supervisor

    def start_pipeline(self, request: StartPipelineRequest, context: grpc.ServicerContext) -> PipelineResponse:
        tid = threading.get_native_id()
        pid = os.getpid()

        logger.info(f"[{pid}][{tid}]: Starting pipeline with request - {str(request)}")
        self._supervisor.start_pipeline(
            json.loads(request.pipeline_config.value),
            request.eval_directory,
            request.start_replay_at,
            request.stop_replay_at,
            request.maximum_triggers,
        )

        # TODO(#317): return pipeline id or something else?
        return PipelineResponse(pipeline_id=1)
