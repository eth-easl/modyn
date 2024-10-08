"""Evaluator GRPC servicer."""

import gc
import logging
import multiprocessing as mp
import pathlib
import queue
import threading
from collections import defaultdict
from threading import Lock

import grpc

from modyn.common.ftp import download_trained_model

# pylint: disable-next=no-name-in-module
from modyn.evaluator.internal.grpc.generated.evaluator_pb2 import (
    EvaluateModelIntervalResponse,
    EvaluateModelRequest,
    EvaluateModelResponse,
    EvaluationAbortedReason,
    EvaluationCleanupRequest,
    EvaluationCleanupResponse,
    EvaluationIntervalData,
    EvaluationResultRequest,
    EvaluationResultResponse,
    EvaluationStatusRequest,
    EvaluationStatusResponse,
    SingleMetricResult,
)
from modyn.evaluator.internal.grpc.generated.evaluator_pb2_grpc import EvaluatorServicer
from modyn.evaluator.internal.pytorch_evaluator import evaluate
from modyn.evaluator.internal.utils import EvaluationInfo, EvaluationProcessInfo
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
        """Initialize evaluator GRPC servicer.

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
        # Note: This only works because the evaluator currently only uses threads, not processes!
        self._evaluation_data_dict: dict[int, defaultdict[int, list[SingleMetricResult]]] = {}
        self._evaluation_data_dict_locks: dict[int, threading.Lock] = {}
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

    # pylint: disable=too-many-locals,too-many-return-statements

    def evaluate_model(self, request: EvaluateModelRequest, context: grpc.ServicerContext) -> EvaluateModelResponse:
        logger.info("Received evaluate model request.")
        num_intervals = len(request.dataset_info.evaluation_intervals)
        with MetadataDatabaseConnection(self._config) as database:
            trained_model: TrainedModel | None = database.session.get(TrainedModel, request.model_id)

            if not trained_model:
                logger.error(f"Trained model {request.model_id} does not exist!")
                return EvaluateModelResponse(
                    evaluation_started=False,
                    interval_responses=[
                        EvaluateModelIntervalResponse(
                            eval_aborted_reason=EvaluationAbortedReason.MODEL_NOT_EXIST_IN_METADATA
                        )
                    ]
                    * num_intervals,
                )
            model_class_name, model_config, amp = database.get_model_configuration(trained_model.pipeline_id)

        if not hasattr(dynamic_module_import("modyn.models"), model_class_name):
            logger.error(f"Model {model_class_name} not available!")
            return EvaluateModelResponse(
                evaluation_started=False,
                interval_responses=[
                    EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.MODEL_IMPORT_FAILURE)
                ]
                * num_intervals,
            )

        fetch_request = FetchModelRequest(model_id=request.model_id, load_metadata=False)
        fetch_resp: FetchModelResponse = self._model_storage_stub.FetchModel(fetch_request)

        if not fetch_resp.success:
            logger.error(
                f"Trained model {request.model_id} cannot be fetched from model storage. "
                f"Evaluation cannot be started."
            )
            return EvaluateModelResponse(
                evaluation_started=False,
                interval_responses=[
                    EvaluateModelIntervalResponse(
                        eval_aborted_reason=EvaluationAbortedReason.MODEL_NOT_EXIST_IN_STORAGE
                    )
                ]
                * num_intervals,
            )

        interval_responses = []
        not_failed_interval_ids: list[int] = []
        for idx, interval in enumerate(request.dataset_info.evaluation_intervals):
            dataset_size_req = GetDatasetSizeRequest(
                dataset_id=request.dataset_info.dataset_id,
                start_timestamp=interval.start_timestamp if interval.HasField("start_timestamp") else None,
                end_timestamp=interval.end_timestamp if interval.HasField("end_timestamp") else None,
            )
            dataset_size_response: GetDatasetSizeResponse = self._storage_stub.GetDatasetSize(dataset_size_req)

            dataset_size = dataset_size_response.num_keys
            if not dataset_size_response.success:
                logger.error(f"The interval {interval} in dataset {request.dataset_info.dataset_id} does not exist.")
                interval_responses.append(
                    EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.DATASET_NOT_FOUND)
                )
            elif dataset_size == 0:
                logger.error(f"The interval {interval} in dataset {request.dataset_info.dataset_id} is empty.")
                interval_responses.append(
                    EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.EMPTY_DATASET)
                )
            else:
                interval_responses.append(
                    EvaluateModelIntervalResponse(
                        eval_aborted_reason=EvaluationAbortedReason.NOT_ABORTED, dataset_size=dataset_size
                    )
                )
                not_failed_interval_ids.append(idx)

        if len(not_failed_interval_ids) == 0:
            logger.error("All evaluations failed. Evaluation cannot be started.")
            return EvaluateModelResponse(evaluation_started=False, interval_responses=interval_responses)

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
            return EvaluateModelResponse(
                evaluation_started=False,
                interval_responses=[
                    EvaluateModelIntervalResponse(eval_aborted_reason=EvaluationAbortedReason.DOWNLOAD_MODEL_FAILURE)
                ]
                * num_intervals,
            )
        evaluation_info = EvaluationInfo(
            request,
            evaluation_id,
            model_class_name,
            model_config,
            amp,
            self._storage_address,
            trained_model_path,
            not_failed_interval_ids,
        )

        self._evaluation_dict[evaluation_id] = evaluation_info
        self._evaluation_data_dict_locks[evaluation_id] = threading.Lock()
        self._evaluation_data_dict[evaluation_id] = defaultdict(list)
        self._run_evaluation(evaluation_id)

        logger.info(f"Started evaluation {evaluation_id}.")
        return EvaluateModelResponse(
            evaluation_started=True,
            evaluation_id=evaluation_id,
            interval_responses=interval_responses,
        )

    def _run_evaluation(self, evaluation_id: int) -> None:
        exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        metric_result_queue: mp.Queue[tuple[str, float]] = mp.Queue()  # pylint: disable=unsubscriptable-object

        process = mp.Process(
            target=evaluate,
            args=(
                self._evaluation_dict[evaluation_id],
                self._base_dir / f"log-{evaluation_id}.txt",
                exception_queue,
                metric_result_queue,
            ),
        )
        process.start()
        self._evaluation_process_dict[evaluation_id] = EvaluationProcessInfo(
            process, exception_queue, metric_result_queue
        )

    def get_evaluation_status(
        self, request: EvaluationStatusRequest, context: grpc.ServicerContext
    ) -> EvaluationStatusResponse:
        evaluation_id = request.evaluation_id
        logger.info(f"Received status request for evaluation {evaluation_id}.")

        if evaluation_id not in self._evaluation_dict:
            logger.error(f"Evaluation with id {evaluation_id} has not been registered")
            return EvaluationStatusResponse(valid=False)

        self._drain_result_queue(evaluation_id)

        process_handler = self._evaluation_process_dict[evaluation_id].process_handler
        if process_handler.is_alive():
            logger.info(f"Evaluation {evaluation_id} is still running.")
            return EvaluationStatusResponse(valid=True, is_running=True)

        exception = self._check_for_evaluation_exception(evaluation_id)
        logger.info(
            f"Evaluation {evaluation_id} is no longer running. "
            f"Process finished {'successfully' if not exception else 'with errors'}."
        )
        return EvaluationStatusResponse(valid=True, is_running=False, exception=exception)

    def _check_for_evaluation_exception(self, evaluation_id: int) -> str | None:
        exception_queue = self._evaluation_process_dict[evaluation_id].exception_queue

        # As qsize() is unreliable and not implemented on macOS,
        # we try to fetch an element within 100ms. If there is no
        # element within that timeframe returned, we return None.
        try:
            exception = exception_queue.get(timeout=0.1)
            return exception
        except queue.Empty:
            return None

    def _drain_result_queue(self, evaluation_id: int) -> None:
        with self._evaluation_data_dict_locks[evaluation_id]:
            metric_result_queue = self._evaluation_process_dict[evaluation_id].metric_result_queue
            while True:
                try:
                    interval_idx, metric_result = metric_result_queue.get(timeout=0.1)
                except queue.Empty:
                    break
                metric_result = [SingleMetricResult(metric=name, result=result) for name, result in metric_result]
                logger.info(
                    f"Got {len(metric_result)} new results for evaluation {evaluation_id} (interval {interval_idx})"
                )
                self._evaluation_data_dict[evaluation_id][interval_idx].extend(metric_result)

        logger.info(f"Drained results queue for evaluation {evaluation_id}")

    def get_evaluation_result(
        self, request: EvaluationResultRequest, context: grpc.ServicerContext
    ) -> EvaluationResultResponse:
        evaluation_id = request.evaluation_id
        logger.info(f"Received get evaluation result request for evaluation {evaluation_id}.")

        if evaluation_id not in self._evaluation_dict:
            logger.error(f"Evaluation {evaluation_id} has not been registered.")
            return EvaluationResultResponse(valid=False)

        self._drain_result_queue(evaluation_id)  # Should already be drained, but just make sure

        if self._evaluation_process_dict[evaluation_id].process_handler.is_alive():
            logger.error(f"Evaluation {evaluation_id} is still running.")
            return EvaluationResultResponse(valid=False)

        logger.info(f"[Evaluation {evaluation_id}] Returning results of all metrics.")
        self._drain_result_queue(evaluation_id)  # Should not do anything, but let's make sure

        evaluation_data: list[EvaluationIntervalData] = []

        for interval_idx, metric_result in self._evaluation_data_dict[evaluation_id].items():
            single_eval_data = EvaluationIntervalData(interval_index=interval_idx, evaluation_data=metric_result)
            evaluation_data.append(single_eval_data)

        expected_results = len(self._evaluation_dict[evaluation_id].not_failed_interval_ids)
        if len(evaluation_data) < expected_results:
            logger.error(
                f"Could not retrieve results for all intervals of evaluation {evaluation_id}. "
                f"Expected {expected_results} results, "
                f"but got {len(evaluation_data)} results. Most likely, an exception happened during evaluation."
            )
            return EvaluationResultResponse(valid=False)

        return EvaluationResultResponse(valid=True, evaluation_results=evaluation_data)

    def cleanup_evaluations(
        self, request: EvaluationCleanupRequest, context: grpc.ServicerContext
    ) -> EvaluationCleanupResponse:
        evaluation_ids = request.evaluation_ids
        logger.info(f"Received cleanup request for evaluations {evaluation_ids}.")

        already_cleaned = [
            evaluation_id for evaluation_id in evaluation_ids if evaluation_id not in self._evaluation_process_dict
        ]
        not_yet_cleaned = [
            evaluation_id for evaluation_id in evaluation_ids if evaluation_id in self._evaluation_process_dict
        ]

        for evaluation_id in not_yet_cleaned:
            process_handler = self._evaluation_process_dict[evaluation_id].process_handler
            if process_handler.is_alive():
                logger.info(f"Clean evaluation {evaluation_id}, which was still running. Cancelling the evaluation.")
                process_handler.terminate()
                process_handler.join(timeout=30)
                if process_handler.is_alive():
                    process_handler.kill()

            self._evaluation_process_dict.pop(evaluation_id)

        for e_id in evaluation_ids:
            if e_id in self._evaluation_dict:
                self._evaluation_dict.pop(e_id)
            if e_id in self._evaluation_data_dict:
                self._evaluation_data_dict.pop(e_id)
            if e_id in self._evaluation_data_dict_locks:
                self._evaluation_data_dict_locks.pop(e_id)

        gc.collect()
        return EvaluationCleanupResponse(succeeded=list(sorted(already_cleaned + not_yet_cleaned)))
