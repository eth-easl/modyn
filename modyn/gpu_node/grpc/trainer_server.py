import json
import grpc
import os
import sys
from pathlib import Path
import logging
import multiprocessing as mp

import yaml

from modyn.gpu_node.grpc.trainer_server_pb2 import (
    RegisterTrainServerRequest,
    RegisterTrainServerResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
    StartTrainingRequest,
    StartTrainingResponse
)

from modyn.gpu_node.utils.model_utils import get_model

from modyn.gpu_node.mocks.mock_selector_server import MockSelectorServer, RegisterTrainingRequest
from modyn.gpu_node.data.utils import prepare_dataloaders

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
        return TrainerAvailableResponse(available=True)

    def register(
        self,
        request: RegisterTrainServerRequest,
        context: grpc.ServicerContext
    ) -> RegisterTrainServerResponse:

        optimizer_dict = json.loads(request.optimizer_parameters.value)
        model_conf_dict = json.loads(request.model_configuration.value)

        train_dataloader, val_dataloader = prepare_dataloaders(
            request.training_id,
            request.data_info.dataset_id,
            request.data_info.num_dataloaders,
            request.batch_size
        )

        if train_dataloader is None:
            return RegisterTrainServerResponse(success=False)

        model = get_model(request, optimizer_dict, model_conf_dict, train_dataloader, val_dataloader, 0)
        self._training_dict[request.training_id] = model

        return RegisterTrainServerResponse(success=True)

    def start_training(self, request: StartTrainingRequest, context: grpc.ServicerContext) -> StartTrainingResponse:

        training_id = request.training_id

        if not training_id in self._training_dict:
            raise ValueError(f"Training with id {training_id} has not been registered")

        if training_id in self._training_process_dict:
            if self._training_process_dict[training_id].is_alive():
                return StartTrainingResponse(training_started=False)

        model = self._training_dict[training_id]

        p = mp.Process(target=model.train, args=(f'log-{training_id}.txt', request.load_checkpoint_path,))
        p.start()
        self._training_process_dict[training_id] = p

        return StartTrainingResponse(training_started=True)
