import glob
import logging
import io
from typing import Any, Optional
import grpc
import os
import sys
from pathlib import Path
import multiprocessing as mp

import torch

logger = logging.getLogger(__name__)

# pylint: disable=no-name-in-module
from modyn.trainer_server.internal.grpc.generated.trainer_server_pb2 import (
    RegisterTrainServerRequest,
    RegisterTrainServerResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    TrainingStatusRequest,
    TrainingStatusResponse
)
from modyn.trainer_server.internal.trainer.pytorch_trainer import train
from modyn.trainer_server.internal.utils.training_info import STATUS_QUERY_MESSAGE, TrainingInfo
from modyn.trainer_server.internal.utils.training_process_info import TrainingProcessInfo


path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


class TrainerServerGRPCServicer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self) -> None:
        self._training_dict: dict[int, TrainingInfo] = {}
        self._training_process_dict: dict[int, TrainingProcessInfo] = {}

    def trainer_available(
        self,
        request: TrainerAvailableRequest,   # pylint: disable=unused-argument
        context: grpc.ServicerContext   # pylint: disable=unused-argument
    ) -> TrainerAvailableResponse:

        # if there is already another training job running, the node is considered unavailable
        for _, training in self._training_process_dict.items():
            if training.process_handler.is_alive():
                return TrainerAvailableResponse(available=False)

        return TrainerAvailableResponse(available=True)

    def register(
        self,
        request: RegisterTrainServerRequest,
        context: grpc.ServicerContext   # pylint: disable=unused-argument
    ) -> RegisterTrainServerResponse:

        training_info = TrainingInfo(request)

        self._training_dict[request.training_id] = training_info

        return RegisterTrainServerResponse(success=True)

    def start_training(
        self,
        request: StartTrainingRequest,
        context: grpc.ServicerContext   # pylint: disable=unused-argument
    ) -> StartTrainingResponse:

        training_id = request.training_id

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered")
            return StartTrainingResponse(training_started=False)

        exception_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_query_queue: mp.Queue[str] = mp.Queue()  # pylint: disable=unsubscriptable-object
        status_response_queue: mp.Queue[dict[str, Any]] = mp.Queue()  # pylint: disable=unsubscriptable-object

        process = mp.Process(
            target=train,
            args=(
                self._training_dict[training_id],
                request.device,
                f'log-{training_id}.txt',
                request.load_checkpoint_path,
                request.train_until_sample_id,
                exception_queue,
                status_query_queue,
                status_response_queue
            )
        )
        process.start()
        self._training_process_dict[training_id] = TrainingProcessInfo(
            process,
            exception_queue,
            status_query_queue,
            status_response_queue
        )

        return StartTrainingResponse(training_started=True)

    def get_training_status(
        self,
        request: TrainingStatusRequest,
        context: grpc.ServicerContext   # pylint: disable=unused-argument
    ) -> TrainingStatusResponse:

        training_id = request.training_id

        if training_id not in self._training_dict:
            logger.error(f"Training with id {training_id} has not been registered")
            return

        process_handler = self._training_process_dict[training_id].process_handler
        if process_handler.is_alive():
            # TODO(fotstrt): what to do if blocked - add a timeout?
            training_state_running, iteration = self.get_status(training_id)
            return TrainingStatusResponse(
                is_running=True,
                state_available=True,
                iteration=iteration,
                state=training_state_running
            )
        exception = self.check_for_training_exception(training_id)
        training_state_finished, iteration = self.get_latest_checkpoint(training_id)
        if exception is None:
            if training_state_finished is not None:
                return TrainingStatusResponse(
                    is_running=False,
                    state_available=True,
                    iteration=iteration,
                    state=training_state_finished
                )
            return TrainingStatusResponse(
                is_running=False,
                state_available=False,
            )
        if training_state_finished is not None:
            return TrainingStatusResponse(
                is_running=False,
                state_available=True,
                exception=exception,
                iteration=iteration,
                state=training_state_finished
            )
        return TrainingStatusResponse(
            is_running=False,
            state_available=False,
            exception=exception,
        )

    def get_status(self, training_id: int) -> tuple[bytes, int]:

        status_query_queue = self._training_process_dict[training_id].status_query_queue
        status_query_queue.put(STATUS_QUERY_MESSAGE)
        response = self._training_process_dict[training_id].status_response_queue.get()
        return response['state'], response['iteration']

    def check_for_training_exception(self, training_id: int) -> Optional[str]:

        exception_queue = self._training_process_dict[training_id].exception_queue
        if exception_queue.qsize() > 0:
            return exception_queue.get()
        return None

    def get_latest_checkpoint(self, training_id: int) -> tuple[Optional[bytes], int]:

        # this might be useful in case that the training has already finished,
        # either successfully or not, and allow to access the last state

        checkpoint_path = self._training_dict[training_id].checkpoint_path
        checkpoints = list(filter(os.path.isfile, glob.glob(checkpoint_path + "/*")))
        checkpoints.sort(key=os.path.getmtime)

        if len(checkpoints) == 0:
            return None, -1

        # TODO(fotstrt): add checks/actions in case checkpoint is corrupted
        checkpoint = checkpoints[-1]
        state = torch.load(checkpoint)
        iteration = state.pop('iteration')
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        state_bytes = buffer.read()
        return state_bytes, iteration
