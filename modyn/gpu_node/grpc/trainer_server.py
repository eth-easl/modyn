import json
import grpc
import os
import sys
from concurrent import futures
from pathlib import Path
import logging
import multiprocessing as mp

import yaml

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import add_TrainerServerServicer_to_server
from modyn.gpu_node.grpc.trainer_server_pb2 import (
    RegisterTrainServerRequest,
    RegisterTrainServerResponse,
    TrainerAvailableRequest,
    TrainerAvailableResponse,
    StartTrainingRequest,
    StartTrainingResponse
)

from modyn.gpu_node.models.utils import get_model

from modyn.backend.selector.mock_selector_server import MockSelectorServer, RegisterTrainingRequest
from modyn.gpu_node.data.utils import prepare_dataloaders

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

logging.basicConfig(format='%(asctime)s %(message)s')


class TrainerGRPCServer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self):
        self._selector = MockSelectorServer()
        self._training_dict = {}
        self._training_process_dict = {}

    def register_with_selector(self, num_dataloaders: int) -> int:
        # TODO: replace this with grpc calls to the selector
        req = RegisterTrainingRequest(num_workers=num_dataloaders)
        response = self._selector.register_training(req)
        return response.training_id

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

        training_id = self.register_with_selector(request.data_info.num_dataloaders)
        print(training_id)

        optimizer_dict = json.loads(request.optimizer_parameters)
        model_conf_dict = json.loads(request.model_configuration)

        train_dataloader, val_dataloader = prepare_dataloaders(
            training_id,
            request.data_info.dataset_id,
            request.data_info.num_dataloaders,
            request.batch_size
        )

        if train_dataloader is None:
            return RegisterTrainServerResponse(training_id=-1)

        model = get_model(request, optimizer_dict, model_conf_dict, train_dataloader, val_dataloader, 0)
        self._training_dict[training_id] = model

        return RegisterTrainServerResponse(training_id=training_id)

    def start_training(self, request: StartTrainingRequest, context: grpc.ServicerContext) -> StartTrainingResponse:

        training_id = request.training_id
        if training_id in self._training_process_dict:
            if self._training_process_dict[training_id].is_alive():
                return StartTrainingResponse(training_started=False)

        model = self._training_dict[training_id]

        p = mp.Process(target=model.train, args=(f'log-{training_id}.txt', request.load_checkpoint_path,))
        p.start()
        self._training_process_dict[training_id] = p

        return StartTrainingResponse(training_started=True)


def serve(config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_TrainerServerServicer_to_server(
        TrainerGRPCServer(), server)
    logging.info(
        'Starting trainer server. Listening on port .' +
        config['trainer']['port'])
    server.add_insecure_port('[::]:' + config['trainer']['port'])
    print("start serving!")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':

    mp.set_start_method('spawn')

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python trainer_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
