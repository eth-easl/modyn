"""Evaluator GRPC servicer."""

import logging
import multiprocessing as mp
import pathlib
import queue
from threading import Lock
from typing import Any, Optional

import grpc
from modyn.common.ftp import download_file, get_pretrained_model_callback

# pylint: disable-next=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationData,
    EvaluationStatusRequest,
    EvaluationStatusResponse,
    FinalEvaluationRequest,
    FinalEvaluationResponse,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2_grpc import EvaluatorServicer
from modyn.evaluator.internal.metric_manager import MetricManager
from modyn.evaluator.internal.metrics import AbstractEvaluationMetric
from modyn.evaluator.internal.pytorch_evaluator import evaluate
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluationProcessInfo, EvaluatorMessages

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.utils import dynamic_module_import, grpc_connection_established

logger = logging.getLogger(__name__)


class EvaluatorGRPCServicer(EvaluatorServicer):
    """GRPC servicer for the evaluator module."""

    def __init__(self, config: dict, tempdir: pathlib.Path):
        """
        Initialize evaluator GRPC servicer.

        Args:
            config: configuration file for evaluator.
            tempdir: temporary directory to store trained models.
        """
        super().__init__()

        self._config = config
        self._base_dir = tempdir

        assert self._base_dir.exists(), f"Temporary Directory {self._base_dir} should have been created."

        self._lock = Lock()

        self._next_evaluation_id = 0
        self._evaluation_dict: dict[int, EvaluationInfo] = {}
        self._evaluation_process_dict: dict[int, EvaluationProcessInfo] = {}

        self._storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
        self._model_storage_address = f"{config['model_storage']['hostname']}:{config['model_storage']['port']}"
        self._metric_manager = MetricManager()

        self._model_storage_stub = EvaluatorGRPCServicer.connect_to_model_storage(self._model_storage_address)

    @staticmethod
    def connect_to_model_storage(model_storage_address: str) -> ModelStorageStub:
        model_storage_channel = grpc.insecure_channel(model_storage_address)
        assert model_storage_channel is not None
        if not grpc_connection_established(model_storage_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to model storage at address {model_storage_address}."
            )
        return ModelStorageStub(model_storage_channel)

    def evaluate_model(self, request: EvaluateModelRequest, context: grpc.ServicerContext) -> EvaluateModelResponse:
        logger.info("Received evaluate model request.")

        if not hasattr(dynamic_module_import("modyn.models"), request.model_id):
            logger.error(f"Model {request.model_id} not available!")
            return EvaluateModelResponse(evaluation_started=False)

        fetch_request = FetchModelRequest(model_id=request.trained_model_id)
        fetch_resp: FetchModelResponse = self._model_storage_stub.FetchModel(fetch_request)

        if not fetch_resp.success:
            logger.error(
                f"Trained model {request.trained_model_id} cannot be fetched from model storage. "
                f"Evaluation cannot be started."
            )
            return EvaluateModelResponse(evaluation_started=False)

        with self._lock:
            evaluation_id = self._next_evaluation_id
            self._next_evaluation_id += 1

        local_model_path = self._base_dir / f"trained_model_{evaluation_id}.modyn"

        download_file(
            hostname=self._config["model_storage"]["hostname"],
            port=int(self._config["model_storage"]["ftp_port"]),
            user="modyn",
            password="modyn",
            remote_file_path=pathlib.Path(fetch_resp.model_path),
            local_file_path=local_model_path,
            callback=get_pretrained_model_callback(logger),
        )

        logger.info(f"Successfully downloaded trained model {request.trained_model_id} to {local_model_path}.")

        metrics = self._setup_metrics(request.metrics)
        evaluation_info = EvaluationInfo(request, evaluation_id, self._storage_address, metrics, local_model_path)
        self._evaluation_dict[evaluation_id] = evaluation_info

        exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_query_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_response_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object

        process = mp.Process(
            target=evaluate,
            args=(
                self._evaluation_dict[evaluation_id],
                self._base_dir / f"log-{evaluation_id}.txt",
                exception_queue,
                status_query_queue,
                status_response_queue,
            ),
        )
        process.start()
        self._evaluation_process_dict[evaluation_id] = EvaluationProcessInfo(
            process, exception_queue, status_query_queue, status_response_queue
        )

        logger.info(f"Started evaluation {evaluation_id}.")
        return EvaluateModelResponse(evaluation_started=True, evaluation_id=evaluation_id)

    def _setup_metrics(self, metrics: list[str]) -> list[AbstractEvaluationMetric]:
        return list(map(self._metric_manager.get_evaluation_metric, set(metrics)))

    def get_evaluation_status(
        self, request: EvaluationStatusRequest, context: grpc.ServicerContext
    ) -> EvaluationStatusResponse:
        evaluation_id = request.evaluation_id
        logger.info(f"Received status request for evaluation {evaluation_id}.")

        if evaluation_id not in self._evaluation_dict:
            logger.error(f"Evaluation with id {evaluation_id} has not been registered")
            return EvaluationStatusResponse(valid=False)

        process_handler = self._evaluation_process_dict[evaluation_id].process_handler
        if process_handler.is_alive():
            logger.info(f"Evaluation {evaluation_id} is still running, obtaining info from running process.")
            num_batches, num_samples = self._get_status(evaluation_id)
            response_kwargs_running: dict[str, Any] = {
                "valid": True,
                "is_running": True,
                "blocked": num_batches is None,
                "state_available": num_batches is not None and num_samples is not None,
                "batches_seen": num_batches,
                "samples_seen": num_samples,
            }
            cleaned_kwargs = {k: v for k, v in response_kwargs_running.items() if v is not None}
            return EvaluationStatusResponse(**cleaned_kwargs)  # type: ignore[arg-type]

        exception = self._check_for_evaluation_exception(evaluation_id)
        logger.info(
            f"Evaluation {evaluation_id} is no longer running. "
            f"Process finished {'successfully' if not exception else 'with errors'}."
        )
        response_kwargs_finished: dict[str, Any] = {
            "valid": True,
            "is_running": False,
            "blocked": False,
            "state_available": False,
            "exception": exception,
        }
        cleaned_kwargs = {k: v for k, v in response_kwargs_finished.items() if v is not None}
        return EvaluationStatusResponse(**cleaned_kwargs)  # type: ignore[arg-type]

    def _get_status(self, evaluation_id: int) -> tuple[Optional[int], Optional[int]]:
        status_query_queue = self._evaluation_process_dict[evaluation_id].status_query_queue
        status_query_queue.put(EvaluatorMessages.STATUS_QUERY_MESSAGE)
        try:
            # blocks for 30 seconds
            response = self._evaluation_process_dict[evaluation_id].status_response_queue.get(timeout=30)
            return response["num_batches"], response["num_samples"]
        except queue.Empty:
            return None, None

    def _check_for_evaluation_exception(self, evaluation_id: int) -> Optional[str]:
        exception_queue = self._evaluation_process_dict[evaluation_id].exception_queue

        # As qsize() is unreliable and not implemented on macOS,
        # we try to fetch an element within 100ms. If there is no
        # element within that timeframe returned, we return None.
        try:
            exception = exception_queue.get(timeout=0.1)
            return exception
        except queue.Empty:
            return None

    def get_final_evaluation(
        self, request: FinalEvaluationRequest, context: grpc.ServicerContext
    ) -> FinalEvaluationResponse:
        evaluation_id = request.evaluation_id
        logger.info(f"Received get final evaluation request for evaluation {evaluation_id}.")

        if evaluation_id not in self._evaluation_dict:
            logger.error(f"Evaluation with id {evaluation_id} has not been registered.")
            return FinalEvaluationResponse(valid=False)

        if self._evaluation_process_dict[evaluation_id].process_handler.is_alive():
            logger.error(f"Evaluation with id {evaluation_id} is still running.")
            return FinalEvaluationResponse(valid=False)

        logger.error("Returning results of all metrics.")
        evaluation_data: list[EvaluationData] = []
        for metric in self._evaluation_dict[evaluation_id].metrics:
            evaluation_data.append(
                EvaluationData(metric=metric.__class__.__name__, result=metric.get_evaluation_result())
            )
        return FinalEvaluationResponse(valid=True, evaluation_data=evaluation_data)
