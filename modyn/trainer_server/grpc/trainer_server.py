# pylint: disable=unused-argument, no-name-in-module, unnecessary-lambda, unsubscriptable-object
import glob
import io
from typing import Any, Optional, Union
import grpc
import os
import sys
from pathlib import Path
import multiprocessing as mp

import torch

from modyn.trainer_server.grpc.generated.trainer_server_pb2 import (
    RegisterTrainServerRequest,
    RegisterTrainServerResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
    StartTrainingRequest,
    StartTrainingResponse,
    TrainingStatusRequest,
    TrainingStatusResponse
)
from modyn.trainer_server.trainer.pytorch_trainer import train

from modyn.trainer_server.utils.training_utils import STATUS_QUERY_MESSAGE, TrainingInfo, TrainingProcessInfo

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


class TrainerGRPCServer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self) -> None:
        self._training_dict: dict[int, TrainingInfo] = {}
        self._training_process_dict: dict[int, TrainingProcessInfo] = {}

    def trainer_available(
        self,
        request: TrainerAvailableRequest,
        context: grpc.ServicerContext
    ) -> TrainerAvailableResponse:

        # if there is already another training job running, the node is considered unavailable
        for _, training in self._training_process_dict.items():
            if training.process_handler.is_alive():
                return TrainerAvailableResponse(available=False)

        return TrainerAvailableResponse(available=True)

    def register(
        self,
        request: RegisterTrainServerRequest,
        context: grpc.ServicerContext
    ) -> RegisterTrainServerResponse:

        training_info = TrainingInfo(request)

        self._training_dict[request.training_id] = training_info

        return RegisterTrainServerResponse(success=True)

    def start_training(self, request: StartTrainingRequest, context: grpc.ServicerContext) -> StartTrainingResponse:

        training_id = request.training_id

        if training_id not in self._training_dict:
            raise ValueError(f"Training with id {training_id} has not been registered")

        exception_queue: mp.Queue[str] = mp.Queue()
        status_query_queue: mp.Queue[str] = mp.Queue()
        status_response_queue: mp.Queue[dict[str, Any]] = mp.Queue()

        process = mp.Process(
            target=train,
            args=(
                self._training_dict[training_id],
                'cuda:0',  # TODO(): fix device number for multi-gpu settings
                f'log-{training_id}.txt',
                request.load_checkpoint_path,
                request.trigger_point,
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
        context: grpc.ServicerContext
    ) -> TrainingStatusResponse:

        training_id = request.training_id

        if training_id not in self._training_dict:
            raise ValueError(f"Training with id {training_id} has not been registered")

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
        exception = self.get_child_exception(training_id)
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

    def get_child_exception(self, training_id: int) -> Union[str, None]:

        exception_queue = self._training_process_dict[training_id].exception_queue
        if exception_queue.qsize() > 0:
            exception_msg = exception_queue.get()
            return exception_msg
        return None

    def get_latest_checkpoint(self, training_id: int) -> tuple[Optional[bytes], int]:

        # this might be useful in case that the training has already finished,
        # either successfully or not, and allow to access the last state

        checkpoint_path = self._training_dict[training_id].checkpoint_path
        checkpoints = list(filter(os.path.isfile, glob.glob(checkpoint_path + "/*")))
        checkpoints.sort(key=lambda x: os.path.getmtime(x))

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
