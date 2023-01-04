import grpc
import os
import sys
from pathlib import Path
import multiprocessing as mp

from modyn.trainer_server.grpc.trainer_server_pb2 import (
    IsRunningRequest,
    IsRunningResponse,
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

from modyn.trainer_server.mocks.mock_selector_server import MockSelectorServer
from modyn.trainer_server.utils.training_utils import TrainingInfo

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


class TrainerGRPCServer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self):
        self._selector_stub = MockSelectorServer()
        self._training_dict = {}
        self._training_process_dict = {}

    def trainer_available(
        self,
        request: TrainerAvailableRequest,
        context: grpc.ServicerContext
    ) -> TrainerAvailableResponse:

        # if there is already another training job running, the node is considered unavailable
        for _, process in self._training_process_dict.items():
            if process.is_alive():
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

        p = mp.Process(
            target=train,
            args=(
                self._training_dict[training_id],
                0,
                f'log-{training_id}.txt',
                request.load_checkpoint_path
            )
        )
        p.start()
        self._training_process_dict[training_id] = p

        return StartTrainingResponse(training_started=True)

    def is_running(self, request: IsRunningRequest, context: grpc.ServicerContext) -> IsRunningResponse:

        training_id = request.training_id
        if training_id in self._training_process_dict:
            process_handler = self._training_process_dict[training_id]
            if process_handler.is_alive():
                return IsRunningResponse(is_running=True)

        return IsRunningResponse(is_running=False)


    def get_training_status(self, request: TrainingStatusRequest, context: grpc.ServicerContext) -> TrainingStatusResponse:

        training_id = request.training_id

        # TODO(fotstrt): send proper response here
        assert training_id in self._training_process_dict

        process_handler = self._training_process_dict[training_id]
        if process_handler.is_alive():
            # case 1
            # TODO(fotstrt): what to do if blocked - add a timeout?
            training_state, iteration = self.get_model(training_id)
            return TrainingStatusResponse(
                is_running=True,
                iteration=iteration,
                state=training_state
            )
        else:
            # case 2
            exception = self.get_child_exception(training_id)
            training_state, iteration = self.get_latest_checkpoint(training_id)
            if exception is None:
                return TrainingStatusResponse(
                    is_running=False,
                    iteration=iteration,
                    state=training_state
                )
            else:
                return TrainingStatusResponse(
                    is_running=False,
                    exception=exception,
                    iteration=iteration,
                    state=training_state
                )


    def get_model(self, training_id):
        # TODO(fotstrt): fill in the actual communication with trainer
        pass

    def get_child_exception(self, training_id):
        # TODO(fotstrt): fill in the actual communication with trainer
        pass

    def get_latest_checkpoint(self, training_id):
        # TODO(fotstrt): find latest checkpoint and load
        pass