"""Evaluator GRPC servicer."""

import json
import logging
import multiprocessing as mp
import pathlib
import queue
from threading import Lock
from typing import Any, Optional

import grpc
from modyn.common.ftp import download_trained_model

# pylint: disable-next=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationData,
    EvaluationResultRequest,
    EvaluationResultResponse,
    EvaluationStatusRequest,
    EvaluationStatusResponse,
    MetricConfiguration,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2_grpc import EvaluatorServicer
from modyn.evaluator.internal.metric_factory import MetricFactory
from modyn.evaluator.internal.metrics import AbstractEvaluationMetric
from modyn.evaluator.internal.pytorch_evaluator import evaluate
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluationProcessInfo, EvaluatorMessages
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import TrainedModel

# pylint: disable-next=no-name-in-module
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import FetchModelRequest, FetchModelResponse
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub

# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import GetDatasetSizeRequest, GetDatasetSizeResponse
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
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
        self._storage_stub = EvaluatorGRPCServicer.connect_to_storage(self._storage_address)

        model_storage_address = f"{config['model_storage']['hostname']}:{config['model_storage']['port']}"
        self._model_storage_stub = EvaluatorGRPCServicer.connect_to_model_storage(model_storage_address)

    @staticmethod
    def connect_to_model_storage(model_storage_address: str) -> ModelStorageStub:
        model_storage_channel = grpc.insecure_channel(model_storage_address)
        assert model_storage_channel is not None
        if not grpc_connection_established(model_storage_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to model storage at address {model_storage_address}."
            )
        return ModelStorageStub(model_storage_channel)

    @staticmethod
    def connect_to_storage(storage_address: str) -> StorageStub:
        storage_channel = grpc.insecure_channel(storage_address)
        assert storage_channel is not None
        if not grpc_connection_established(storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at address {storage_address}.")
        return StorageStub(storage_channel)

    # pylint: disable=too-many-locals, too-many-return-statements

    def evaluate_model(self, request: EvaluateModelRequest, context: grpc.ServicerContext) -> EvaluateModelResponse:
        logger.info("Received evaluate model request.")

        with MetadataDatabaseConnection(self._config) as database:
            trained_model: Optional[TrainedModel] = database.session.get(TrainedModel, request.model_id)

            if not trained_model:
                logger.error(f"Trained model {request.model_id} does not exist!")
                return EvaluateModelResponse(evaluation_started=False)
            model_class_name, model_config, amp = database.get_model_configuration(trained_model.pipeline_id)

        if not hasattr(dynamic_module_import("modyn.models"), model_class_name):
            logger.error(f"Model {model_class_name} not available!")
            return EvaluateModelResponse(evaluation_started=False)

        fetch_request = FetchModelRequest(model_id=request.model_id, load_metadata=False)
        fetch_resp: FetchModelResponse = self._model_storage_stub.FetchModel(fetch_request)

        if not fetch_resp.success:
            logger.error(
                f"Trained model {request.model_id} cannot be fetched from model storage. "
                f"Evaluation cannot be started."
            )
            return EvaluateModelResponse(evaluation_started=False)

        dataset_info = request.dataset_info
        dataset_size_req = GetDatasetSizeRequest(
            dataset_id=request.dataset_info.dataset_id,
            start_timestamp=dataset_info.start_timestamp,
            end_timestamp=dataset_info.end_timestamp,
        )
        dataset_size_response: GetDatasetSizeResponse = self._storage_stub.GetDatasetSize(dataset_size_req)

        dataset_size = dataset_size_response.num_keys

        if not dataset_size_response.success:
            logger.error(
                f"Total number of keys for dataset {dataset_size_req.dataset_id} cannot be fetched. "
                f"Evaluation cannot be started."
            )
            return EvaluateModelResponse(evaluation_started=False)

        if dataset_size == 0:
            logger.error(
                f"Dataset {dataset_size_req.dataset_id} is empty in given time interval. Evaluation cannot be started."
            )
            return EvaluateModelResponse(evaluation_started=False)

        with self._lock:
            evaluation_id = self._next_evaluation_id
            self._next_evaluation_id += 1

        trained_model_path = download_trained_model(
            logger=logger,
            model_storage_config=self._config["model_storage"],
            remote_path=pathlib.Path(fetch_resp.model_path),
            checksum=fetch_resp.checksum,
            identifier=evaluation_id,
            base_directory=self._base_dir,
        )

        if not trained_model_path:
            logger.error("Trained model could not be downloaded. Evaluation cannot be started.")
            return EvaluateModelResponse(evaluation_started=False)

        metrics = self._setup_metrics(request.metrics)
        evaluation_info = EvaluationInfo(
            request,
            evaluation_id,
            model_class_name,
            model_config,
            amp,
            self._storage_address,
            metrics,
            trained_model_path,
        )

        self._evaluation_dict[evaluation_id] = evaluation_info
        self._run_evaluation(evaluation_id)

        logger.info(f"Started evaluation {evaluation_id}.")
        return EvaluateModelResponse(evaluation_started=True, evaluation_id=evaluation_id, dataset_size=dataset_size)

    @staticmethod
    def _setup_metrics(metric_configurations: list[MetricConfiguration]) -> list[AbstractEvaluationMetric]:
        metrics = []
        # need to make sure that the metric names are unique as they are used for identification.
        metric_names = set()
        for configuration in metric_configurations:
            loaded_config = json.loads(configuration.config.value)
            metric = MetricFactory.get_evaluation_metric(
                configuration.name, configuration.evaluation_transformer.value, loaded_config
            )
            if metric.get_name() not in metric_names:
                metrics.append(metric)
                metric_names.add(metric.get_name())
            else:
                logger.warning(f"Metric {metric.get_name()} is already registered.")
        return metrics

    def _run_evaluation(self, evaluation_id: int) -> None:
        exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_query_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_response_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object
        metric_result_queue: mp.Queue[tuple[str, float]] = mp.Queue()  # pylint: disable=unsubscriptable-object

        process = mp.Process(
            target=evaluate,
            args=(
                self._evaluation_dict[evaluation_id],
                self._base_dir / f"log-{evaluation_id}.txt",
                exception_queue,
                status_query_queue,
                status_response_queue,
                metric_result_queue,
            ),
        )
        process.start()
        self._evaluation_process_dict[evaluation_id] = EvaluationProcessInfo(
            process, exception_queue, status_query_queue, status_response_queue, metric_result_queue
        )

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

    def get_evaluation_result(
        self, request: EvaluationResultRequest, context: grpc.ServicerContext
    ) -> EvaluationResultResponse:
        evaluation_id = request.evaluation_id
        logger.info(f"Received get evaluation result request for evaluation {evaluation_id}.")

        if evaluation_id not in self._evaluation_dict:
            logger.error(f"Evaluation with id {evaluation_id} has not been registered.")
            return EvaluationResultResponse(valid=False)

        if self._evaluation_process_dict[evaluation_id].process_handler.is_alive():
            logger.error(f"Evaluation with id {evaluation_id} is still running.")
            return EvaluationResultResponse(valid=False)

        logger.info("Returning results of all metrics.")
        evaluation_data: list[EvaluationData] = []

        metric_result_queue = self._evaluation_process_dict[evaluation_id].metric_result_queue
        metric_results: list[tuple[str, float]] = []
        for _ in range(len(self._evaluation_dict[evaluation_id].metrics)):
            try:
                metric_results.append(metric_result_queue.get(timeout=0.1))
            except queue.Empty:
                logger.error(f"Evaluation with id {evaluation_id} did not return all metric results.")
                break

        for name, result in metric_results:
            evaluation_data.append(EvaluationData(metric=name, result=result))
        return EvaluationResultResponse(valid=True, evaluation_data=evaluation_data)
