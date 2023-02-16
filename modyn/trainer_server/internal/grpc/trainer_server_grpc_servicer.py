import io
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import sys
import tempfile
import shutil
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import grpc
import torch

# pylint: disable=no-name-in-module
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    GetFinalModelRequest,
    GetFinalModelResponse,
    GetLatestModelRequest,
    GetLatestModelResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
    TrainingStatusRequest,
    TrainingStatusResponse,
)
from modyn.trainer_server.internal.trainer.pytorch_trainer import train
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo
from modyn.trainer_server.internal.utils.training_process_info import TrainingProcessInfo
from modyn.utils.utils import dynamic_module_import

logger = logging.getLogger(__name__)

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


class TrainerServerGRPCServicer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self, config: dict) -> None:
        self._next_training_id = 0
        self._lock = Lock()  # TODO(#118): Fix race conditions in the trainer server
        self._training_dict: dict[int, TrainingInfo] = {}
        self._training_process_dict: dict[int, TrainingProcessInfo] = {}
        self._modyn_base_dir = pathlib.Path(tempfile.gettempdir()) / "modyn"
        
        if self._modyn_base_dir.exists() and self._modyn_base_dir.is_dir():
            shutil.rmtree(self._modyn_base_dir)

        self._modyn_base_dir.mkdir()

        self._storage_address = f"{config['storage']['hostname']}:{config['storage']['port']}"
        self._selector_address = f"{config['selector']['hostname']}:{config['selector']['port']}"

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

    def start_training(
        self,
        request: StartTrainingRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> StartTrainingResponse:
        if not hasattr(dynamic_module_import("modyn.models"), request.model_id):
            logger.error(f"Model {request.model_id} not available!")
            return StartTrainingResponse(training_started=False)

        with self._lock:
            training_id = self._next_training_id
            self._next_training_id += 1

        final_checkpoint_path = self._modyn_base_dir / f"training_{training_id}"
        training_info = TrainingInfo(
            request, training_id, self._storage_address, self._selector_address, final_checkpoint_path
        )
        self._training_dict[training_id] = training_info

        exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_query_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_response_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object

        process = mp.Process(
            target=train,
            args=(
                self._training_dict[training_id],
                request.device,
                self._modyn_base_dir / f"log-{training_id}.txt",
                exception_queue,
                status_query_queue,
                status_response_queue,
            ),
        )
        process.start()
        self._training_process_dict[training_id] = TrainingProcessInfo(
            process, exception_queue, status_query_queue, status_response_queue
        )

        return StartTrainingResponse(training_started=True, training_id=training_id)

    def get_training_status(
        self,
        request: TrainingStatusRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> TrainingStatusResponse:
        training_id = request.training_id

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered")
            return TrainingStatusResponse(valid=False)

        process_handler = self._training_process_dict[training_id].process_handler
        if process_handler.is_alive():
            _, num_batches, num_samples = self.get_status(training_id)
            response_kwargs_running: dict[str, Any] = {
                "valid": True,
                "is_running": True,
                "blocked": num_batches is None,
                "state_available": num_batches is not None,
                "batches_seen": num_batches,
                "samples_seen": num_samples,
            }
            cleaned_kwargs = {k: v for k, v in response_kwargs_running.items() if v}
            return TrainingStatusResponse(**cleaned_kwargs)  # type: ignore[arg-type]

        exception = self.check_for_training_exception(training_id)
        _, num_batches, num_samples = self.get_latest_checkpoint(training_id)
        response_kwargs_finished: dict[str, Any] = {
            "valid": True,
            "is_running": False,
            "blocked": False,
            "state_available": num_batches is not None,
            "exception": exception,
            "batches_seen": num_batches,
            "samples_seen": num_samples,
        }
        cleaned_kwargs = {k: v for k, v in response_kwargs_finished.items() if v}
        return TrainingStatusResponse(**cleaned_kwargs)  # type: ignore[arg-type]

    def get_final_model(
        self,
        request: GetFinalModelRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> GetFinalModelResponse:
        training_id = request.training_id

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered")
            return GetFinalModelResponse(valid_state=False)

        if self._training_process_dict[training_id].process_handler.is_alive():
            logger.error(f"Training with id {training_id} is still running")
            return GetFinalModelResponse(valid_state=False)

        final_checkpoint_path = self._training_dict[training_id].final_checkpoint_path / "model_final.modyn"
        if final_checkpoint_path.exists():
            final_state = torch.load(final_checkpoint_path)
            buffer = io.BytesIO()
            torch.save(final_state, buffer)
            buffer.seek(0)
            state_bytes = buffer.read()
            return GetFinalModelResponse(valid_state=True, state=state_bytes)
        logger.error(f"Could not find final checkpoint of training with ID {training_id}.")
        return GetFinalModelResponse(valid_state=False)

    def get_latest_model(
        self,
        request: GetLatestModelRequest,
        context: grpc.ServicerContext,  # pylint: disable=unused-argument
    ) -> GetLatestModelResponse:
        training_id = request.training_id

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered")
            return GetLatestModelResponse(valid_state=False)

        process_handler = self._training_process_dict[training_id].process_handler
        if process_handler.is_alive():
            training_state, _, _ = self.get_status(training_id)
        else:
            training_state, _, _ = self.get_latest_checkpoint(training_id)

        if training_state is not None:
            return GetLatestModelResponse(valid_state=True, state=training_state)
        return GetLatestModelResponse(valid_state=False)

    def get_status(self, training_id: int) -> tuple[Optional[bytes], Optional[int], Optional[int]]:
        status_query_queue = self._training_process_dict[training_id].status_query_queue
        status_query_queue.put(TrainerMessages.STATUS_QUERY_MESSAGE)
        try:
            # blocks for 30 seconds
            response = self._training_process_dict[training_id].status_response_queue.get(timeout=30)
            return response["state"], response["num_batches"], response["num_samples"]
        except queue.Empty:
            return None, None, None

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

    def get_latest_checkpoint(self, training_id: int) -> tuple[Optional[bytes], Optional[int], Optional[int]]:
        # this might be useful in case that the training has already finished,
        # either successfully or not, and allow to access the last state

        checkpoint_path = self._training_dict[training_id].checkpoint_path
        if not checkpoint_path.exists():
            return None, None, None

        checkpoints = list(checkpoint_path.iterdir())
        checkpoints.sort(key=os.path.getmtime)

        # get latest valid checkpoint
        for checkpoint in checkpoints:
            try:
                state = torch.load(checkpoint)
                num_batches = state.pop("num_batches")
                num_samples = state.pop("num_samples")
                buffer = io.BytesIO()
                torch.save(state, buffer)
                buffer.seek(0)
                state_bytes = buffer.read()
                return state_bytes, num_batches, num_samples
            except Exception as exception:  # pylint: disable=broad-except
                # checkpoint corrupted
                logger.error(exception)
        return None, None, None
