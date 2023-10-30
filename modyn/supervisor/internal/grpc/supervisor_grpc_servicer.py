import json
import logging
import os
import threading
from typing import Iterable

import grpc

# pylint: disable=no-name-in-module
from modyn.supervisor.internal.grpc.generated.supervisor_pb2 import (
    StartPipelineRequest,
    PipelineResponse,
)
from modyn.supervisor.internal.grpc.generated.supervisor_pb2_grpc import SupervisorServicer  # noqa: E402, E501
from modyn.utils import seed_everything
from modyn.supervisor import Supervisor

logger = logging.getLogger(__name__)


class SupervisorGRPCServicer(SupervisorServicer):
    """Provides methods that implement functionality of the selector."""

    def __init__(self, modyn_config) -> None:
        self._supervisor = Supervisor(modyn_config)

    def start_pipeline(self, request: StartPipelineRequest, context: grpc.ServicerContext) -> PipelineResponse:
        logger.info(f"Starting pipeline with request - {str(request)}")
        self._supervisor.start_pipeline(
            json.loads(request.pipeline_config.value), request.eval_directory, request.start_replay_at, request.stop_replay_at, request.maximum_triggers 
        )