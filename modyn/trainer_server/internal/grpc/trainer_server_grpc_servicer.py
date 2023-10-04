import logging
import multiprocessing as mp
import os
import pathlib
import queue
from threading import Lock
from typing import Any, Optional, Tuple, Union

import grpc
import torch

# pylint: disable=no-name-in-module
from modyn.common.ftp import download_file, get_pretrained_model_callback
from modyn.model_storage.internal.grpc.generated.model_storage_pb2 import (
    FetchModelRequest,
    FetchModelResponse,
    RegisterModelRequest,
    RegisterModelResponse,
)
from modyn.model_storage.internal.grpc.generated.model_storage_pb2_grpc import ModelStorageStub
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    GetLatestModelRequest,
    GetLatestModelResponse,
    JsonString,
    StartTrainingRequest,
    StartTrainingResponse,
    StoreFinalModelRequest,
    StoreFinalModelResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
    TrainingStatusRequest,
    TrainingStatusResponse,
)
from modyn.trainer_server.internal.trainer.pytorch_trainer import train
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo
from modyn.trainer_server.internal.utils.training_process_info import TrainingProcessInfo
from modyn.utils import current_time_millis, dynamic_module_import, grpc_connection_established

logger = logging.getLogger(__name__)


class TrainerServerGRPCServicer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self, config: dict, tempdir: Union[str, pathlib.Path]) -> None:
        self._config = config

        self._next_training_id = 0
        self._lock = Lock()  # TODO(#118): Fix race conditions in the trainer server
        self._training_dict: dict[int, TrainingInfo] = {}
        self._training_process_dict: dict[int, TrainingProcessInfo] = {}
        self._modyn_base_dir = pathlib.Path(tempdir)

        assert self._modyn_base_dir.exists(), f"Temporary Directory {self._modyn_base_dir} should have been created."

        self._tmp_state_dir = self._modyn_base_dir / "tmp-state"
        os.mkdir(self._tmp_state_dir)

        self._storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
        self._selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"
        self.model_storage_stub = TrainerServerGRPCServicer.connect_to_model_storage(
            f"{config['model_storage']['hostname']}:{config['model_storage']['port']}"
        )
        self._offline_dataset_directory = self._config["trainer_server"]["offline_dataset_directory"]
        logger.info("TrainerServer gRPC Servicer initialized.")

    @staticmethod
    def connect_to_model_storage(model_storage_address: str) -> ModelStorageStub:
        model_storage_channel = grpc.insecure_channel(model_storage_address)
        assert model_storage_channel is not None
        if not grpc_connection_established(model_storage_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to model storage at address {model_storage_address}."
            )
        return ModelStorageStub(model_storage_channel)

    def trainer_available(
        self,
        request: TrainerAvailableRequest,  # pylint: disable=unused-argument
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> TrainerAvailableResponse:
        # if there is already another training job running, the node is considered unavailable
        for _, training in self._training_process_dict.items():
            if training.process_handler.is_alive():
                return TrainerAvailableResponse(available=False)

        return TrainerAvailableResponse(available=True)

    # pylint: disable=too-many-locals
    def start_training(
        self,
        request: StartTrainingRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> StartTrainingResponse:
        logger.info("Received start training request.")

        if not hasattr(dynamic_module_import("modyn.models"), request.model_id):
            logger.error(f"Model {request.model_id} not available!")
            return StartTrainingResponse(training_started=False)

        pretrained_model_path: Optional[pathlib.Path] = None
        if request.use_pretrained_model:
            fetch_request = FetchModelRequest(model_id=request.pretrained_model_id)
            fetch_resp: FetchModelResponse = self.model_storage_stub.FetchModel(fetch_request)

            if not fetch_resp.success:
                logger.error(
                    f"Pretrained Model {request.pretrained_model_id} cannot be fetched from model storage. "
                    f"Training cannot be started."
                )
                return StartTrainingResponse(training_started=False)

            with self._lock:
                training_id = self._next_training_id
                self._next_training_id += 1

            pretrained_model_path = self._modyn_base_dir / pathlib.Path(f"pretrained_model_{training_id}.modyn")

            download_file(
                hostname=self._config["model_storage"]["hostname"],
                port=int(self._config["model_storage"]["ftp_port"]),
                user="modyn",
                password="modyn",
                remote_file_path=pathlib.Path(fetch_resp.model_path),
                local_file_path=pretrained_model_path,
                callback=get_pretrained_model_callback(logger),
            )

            logger.info(f"Completed pretrained model download. Local path: {pretrained_model_path}")
        else:
            with self._lock:
                training_id = self._next_training_id
                self._next_training_id += 1

        final_checkpoint_path = self._modyn_base_dir / f"training_{training_id}"
        logfile_path = self._modyn_base_dir / f"training_{training_id}_logs.log"
        training_info = TrainingInfo(
            request,
            training_id,
            self._storage_address,
            self._selector_address,
            self._offline_dataset_directory,
            final_checkpoint_path,
            logfile_path,
            pretrained_model_path=pretrained_model_path,
        )
        self._training_dict[training_id] = training_info

        exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_query_queue_training: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_response_queue_training: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object

        status_query_queue_downsampling: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        # pylint: disable-next=unsubscriptable-object
        status_response_queue_downsampling: mp.Queue[dict[str, Any]] = mp.Queue()

        process = mp.Process(
            target=train,
            args=(
                self._training_dict[training_id],
                request.device,
                self._modyn_base_dir / f"log-{training_id}.txt",
                exception_queue,
                status_query_queue_training,
                status_response_queue_training,
                status_query_queue_downsampling,
                status_response_queue_downsampling,
            ),
        )
        process.start()
        self._training_process_dict[training_id] = TrainingProcessInfo(
            process,
            exception_queue,
            status_query_queue_training,
            status_response_queue_training,
            status_query_queue_downsampling,
            status_response_queue_downsampling,
        )

        logger.info(f"Started training {training_id}")
        return StartTrainingResponse(training_started=True, training_id=training_id)

    # pylint: disable-next=too-many-locals
    def get_training_status(
        self,
        request: TrainingStatusRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> TrainingStatusResponse:
        training_id = request.training_id
        logger.info(f"Received status request for training {training_id}")

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered")
            return TrainingStatusResponse(valid=False)

        process_handler = self._training_process_dict[training_id].process_handler
        if process_handler.is_alive():
            logger.info(f"Training {training_id} is still running, obtaining info from running process.")
            (
                is_training,
                downsampling_num_batches,
                downsampling_num_samples,
                training_num_batches,
                training_num_samples,
            ) = self.get_values_from_queues(training_id)

            response_kwargs_running: dict[str, Any] = {
                "valid": True,
                "is_running": True,
                "is_training": is_training,
                "blocked": training_num_batches is None and downsampling_num_batches is None,
                "state_available": (training_num_batches is not None and training_num_samples is not None)
                or (downsampling_num_batches is not None and downsampling_num_samples is not None),
                "batches_seen": training_num_batches,
                "samples_seen": training_num_samples,
                "downsampling_batches_seen": downsampling_num_batches,
                "downsampling_samples_seen": downsampling_num_samples,
            }
            cleaned_kwargs = {k: v for k, v in response_kwargs_running.items() if v is not None}
            return TrainingStatusResponse(**cleaned_kwargs)  # type: ignore[arg-type]

        exception = self.check_for_training_exception(training_id)
        _, num_batches, num_samples = self.get_latest_checkpoint(training_id)
        log_str = self.get_training_log(training_id)

        response_kwargs_finished: dict[str, Any] = {
            "valid": True,
            "is_running": False,
            "is_training": False,
            "blocked": False,
            "state_available": num_batches is not None and num_samples is not None,
            "exception": exception,
            "batches_seen": num_batches,
            "samples_seen": num_samples,
            "log": JsonString(value=log_str),
        }
        cleaned_kwargs = {k: v for k, v in response_kwargs_finished.items() if v is not None}
        return TrainingStatusResponse(**cleaned_kwargs)  # type: ignore[arg-type]

    def get_training_log(self, training_id: int) -> str:
        if "PYTEST_CURRENT_TEST" in os.environ and self._training_dict[training_id] is None:
            return ""  # Simplifies a lot of tests that don't need a training object

        log_file_path = self._training_dict[training_id].log_file_path
        if not log_file_path.is_file():
            logger.error(f"Log File for training {training_id} does not exist at {log_file_path}")
            return ""

        with open(log_file_path, "r", encoding="utf-8") as logfile:
            return logfile.read()

    def get_values_from_queues(
        self, training_id: int
    ) -> Tuple[Optional[bool], Optional[int], Optional[int], Optional[int], Optional[int]]:
        was_training = self._training_process_dict[training_id].was_training

        if was_training:
            (
                is_training,
                downsampling_num_batches,
                downsampling_num_samples,
                training_num_batches,
                training_num_samples,
            ) = self._handle_was_training(training_id)
        else:
            (
                is_training,
                downsampling_num_batches,
                downsampling_num_samples,
                training_num_batches,
                training_num_samples,
            ) = self._handle_was_not_training(training_id)

        if is_training is not None:
            self._training_process_dict[training_id].was_training = is_training

        return (
            is_training,
            downsampling_num_batches,
            downsampling_num_samples,
            training_num_batches,
            training_num_samples,
        )

    def _handle_was_not_training(
        self, training_id: int
    ) -> Tuple[Optional[bool], Optional[int], Optional[int], Optional[int], Optional[int]]:
        downsampling_num_batches, downsampling_num_samples, is_training = self.get_status_downsampling(
            training_id, timeout=15
        )
        if is_training:
            # clean the downsampling queue (downsampling is ended)
            self.clean_downsampling_queue(training_id)
        # read the second queue only if the first one is empty
        if downsampling_num_batches is None:
            training_num_batches, training_num_samples, is_training = self.get_status_training(training_id, timeout=15)
        else:
            training_num_batches, training_num_samples = None, None
        return (
            is_training,
            downsampling_num_batches,
            downsampling_num_samples,
            training_num_batches,
            training_num_samples,
        )

    def _handle_was_training(
        self, training_id: int
    ) -> Tuple[Optional[bool], Optional[int], Optional[int], Optional[int], Optional[int]]:
        training_num_batches, training_num_samples, is_training = self.get_status_training(training_id, timeout=15)
        if not is_training:
            # clean the training queue (epoch is finished)
            self.clean_training_queue(training_id)
        # read the second queue only if the first one is empty
        if training_num_batches is None:
            downsampling_num_batches, downsampling_num_samples, is_training = self.get_status_downsampling(
                training_id, timeout=15
            )
        else:
            downsampling_num_batches, downsampling_num_samples = None, None
        return (
            is_training,
            downsampling_num_batches,
            downsampling_num_samples,
            training_num_batches,
            training_num_samples,
        )

    def store_final_model(
        self,
        request: StoreFinalModelRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> StoreFinalModelResponse:
        training_id = request.training_id
        logger.info(f"Received get final model request for training {training_id}.")

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered.")
            return StoreFinalModelResponse(valid_state=False)

        if self._training_process_dict[training_id].process_handler.is_alive():
            logger.error(f"Training with id {training_id} is still running.")
            return StoreFinalModelResponse(valid_state=False)

        final_model_path = self._training_dict[training_id].final_checkpoint_path / "model_final.modyn"
        if final_model_path.exists():
            prefix_path = str(final_model_path.relative_to(self._modyn_base_dir))

            pipeline_id = self._training_dict[training_id].pipeline_id
            trigger_id = self._training_dict[training_id].trigger_id

            register_request = RegisterModelRequest(
                pipeline_id=pipeline_id,
                trigger_id=trigger_id,
                hostname=self._config["trainer_server"]["hostname"],
                port=int(self._config["trainer_server"]["ftp_port"]),
                model_path=prefix_path,
            )

            register_response: RegisterModelResponse = self.model_storage_stub.RegisterModel(register_request)

            if not register_response.success:
                logger.error(f"Could not store final model from training id {training_id}.")
                return StoreFinalModelResponse(valid_state=False)

            os.remove(final_model_path)

            logger.info(f"Deleted final model at {final_model_path}")

            return StoreFinalModelResponse(valid_state=True, model_id=register_response.model_id)

        logger.error(f"Could not find final checkpoint of training with ID {training_id}.")
        return StoreFinalModelResponse(valid_state=False)

    def get_latest_model(
        self,
        request: GetLatestModelRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> GetLatestModelResponse:
        training_id = request.training_id
        logger.info(f"Received get latest model request for training {training_id}.")

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered")
            return GetLatestModelResponse(valid_state=False)

        process_handler = self._training_process_dict[training_id].process_handler
        if process_handler.is_alive():
            training_state = self.get_model_state(training_id)
            if training_state is not None:
                checkpoint_path = self._modyn_base_dir / "tmp-state" / f"{current_time_millis()}-{training_id}"
                TrainerServerGRPCServicer.persist_state_to_disk(training_state, checkpoint_path)
            else:
                checkpoint_path = None
        else:
            checkpoint_path, _, _ = self.get_latest_checkpoint(training_id)

        if checkpoint_path is not None:
            prefix_path = str(checkpoint_path.relative_to(self._modyn_base_dir))
            return GetLatestModelResponse(valid_state=True, model_path=prefix_path)
        return GetLatestModelResponse(valid_state=False)

    def get_status_training(
        self, training_id: int, timeout: float = 30
    ) -> tuple[Optional[int], Optional[int], Optional[bool]]:
        status_query_queue = self._training_process_dict[training_id].status_query_queue_training
        status_query_queue.put(TrainerMessages.STATUS_QUERY_MESSAGE)
        try:
            # blocks for timeout seconds
            response = self._training_process_dict[training_id].status_response_queue_training.get(timeout=timeout)
            return response["num_batches"], response["num_samples"], response["training_active"]
        except queue.Empty:
            return None, None, None

    def clean_training_queue(self, training_id: int) -> None:
        training_queue = self._training_process_dict[training_id].status_response_queue_training
        while not training_queue.empty():
            # blocks for 30 seconds
            _ = training_queue.get()

    def clean_downsampling_queue(self, training_id: int) -> None:
        downsampling_queue = self._training_process_dict[training_id].status_response_queue_downsampling
        while not downsampling_queue.empty():
            # blocks for 30 seconds
            _ = downsampling_queue.get()

    def get_status_downsampling(
        self, training_id: int, timeout: float
    ) -> tuple[Optional[int], Optional[int], Optional[bool]]:
        status_query_queue = self._training_process_dict[training_id].status_query_queue_downsampling
        status_query_queue.put(TrainerMessages.STATUS_QUERY_MESSAGE)
        try:
            # blocks for timeout seconds
            response = self._training_process_dict[training_id].status_response_queue_downsampling.get(timeout=timeout)
            return response["num_batches"], response["num_samples"], response["training_active"]
        except queue.Empty:
            return None, None, None

    def get_model_state(self, training_id: int) -> Optional[bytes]:
        status_query_queue = self._training_process_dict[training_id].status_query_queue_training
        status_query_queue.put(TrainerMessages.MODEL_STATE_QUERY_MESSAGE)
        try:
            # blocks for timeout seconds
            response = self._training_process_dict[training_id].status_response_queue_training.get(timeout=30)
            return response
        except queue.Empty:
            return None

    def check_for_training_exception(self, training_id: int) -> Optional[str]:
        exception_queue = self._training_process_dict[training_id].exception_queue

        # As qsize() is unreliable and not implemented on macOS,
        # we try to fetch an element within 100ms. If there is no
        # element within that timeframe returned, we return None.
        try:
            exception = exception_queue.get(timeout=0.1)
            return exception
        except queue.Empty:
            return None

    def get_latest_checkpoint(self, training_id: int) -> tuple[Optional[pathlib.Path], Optional[int], Optional[int]]:
        # this might be useful in case that the training has already finished,
        # either successfully or not, and allow to access the last state

        checkpoint_path = self._training_dict[training_id].checkpoint_path
        checkpoint_interval = self._training_dict[training_id].checkpoint_interval
        if not checkpoint_path.exists() or checkpoint_path == pathlib.Path("") or checkpoint_interval == 0:
            return None, None, None

        checkpoints = list(checkpoint_path.iterdir())
        checkpoints.sort(key=os.path.getmtime)

        # get latest valid checkpoint
        for checkpoint in checkpoints:
            try:
                state = torch.load(checkpoint)
                num_batches = state.pop("num_batches")
                num_samples = state.pop("num_samples")

                return checkpoint, num_batches, num_samples
            except Exception as exception:  # pylint: disable=broad-except
                # checkpoint corrupted
                logger.error(f"The checkpoint {checkpoint} is corrupted: {exception}")
        return None, None, None

    @staticmethod
    def persist_state_to_disk(state: bytes, path: pathlib.Path) -> None:
        with open(path, "wb") as file:
            file.write(state)
