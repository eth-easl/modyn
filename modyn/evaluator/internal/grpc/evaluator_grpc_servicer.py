"""Evaluator GRPC servicer."""

import logging
import pathlib

import grpc

# pylint: disable-next=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationStatusRequest,
    EvaluationStatusResponse,
    FinalEvaluationRequest,
    FinalEvaluationResponse,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2_grpc import EvaluatorServicer

logger = logging.getLogger(__name__)


class EvaluatorGRPCServicer(EvaluatorServicer):
    """GRPC servicer for the storage module."""

    def __init__(self, config: dict, tempdir: pathlib.Path):
        """Initialize the evaluator GRPC servicer.

        Args:
            config (dict): Configuration of the evaluator module.
        """
        super().__init__()

        self._config = config
        self.tempdir = tempdir

    def evaluate_model(self, request: EvaluateModelRequest, context: grpc.ServicerContext) -> EvaluateModelResponse:
        return EvaluateModelResponse()

    def get_evaluation_status(
        self, request: EvaluationStatusRequest, context: grpc.ServicerContext
    ) -> EvaluationStatusResponse:
        return EvaluationStatusResponse()

    def get_final_evaluation(
        self, request: FinalEvaluationRequest, context: grpc.ServicerContext
    ) -> FinalEvaluationResponse:
        return FinalEvaluationResponse()
