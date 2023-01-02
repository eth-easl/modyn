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

        optimizer_dict = json.loads(request.optimizer_parameters.value)
        model_conf_dict = json.loads(request.model_configuration.value)

        # TODO(fotstrt): if we are keeping this way of passing transforms,
        # find a clearer way to pass this (mp.spawn complains on proto structs)
        transform_list = []
        for x in request.transform_list:
            transform_list.append({'function': x.function, 'args': x.args.value})

        train_dataloader, val_dataloader = prepare_dataloaders(
            request.training_id,
            request.data_info.dataset_id,
            request.data_info.num_dataloaders,
            request.batch_size,
            transform_list
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

        model = self._training_dict[training_id]

        p = mp.Process(target=model.train, args=(f'log-{training_id}.txt', request.load_checkpoint_path,))
        p.start()
        self._training_process_dict[training_id] = p

        return StartTrainingResponse(training_started=True)
