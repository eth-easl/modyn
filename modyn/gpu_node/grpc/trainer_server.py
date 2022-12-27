import grpc
import os
import sys
from concurrent import futures
from pathlib import Path
import logging
import torch
import multiprocessing as mp
import importlib

import yaml

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import add_TrainerServerServicer_to_server
from modyn.gpu_node.grpc.trainer_server_pb2 import TrainerServerRequest, TrainerServerResponse, TrainerAvailableRequest, TrainerAvailableResponse
import modyn.models as available_models

from modyn.gpu_node.models.utils import get_model

logging.basicConfig(format='%(asctime)s %(message)s')

from modyn.backend.selector.mock_selector_server import MockSelectorServer, RegisterTrainingRequest
from modyn.gpu_node.data.utils import prepare_dataloaders
from modyn.gpu_node.grpc.server_utils import process_complex_messages

class TrainerGRPCServer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self):
        self._selector = MockSelectorServer()

    def register_with_selector(self, num_dataloaders):
        # TODO: replace this with grpc calls to the selector
        req = RegisterTrainingRequest(num_workers=num_dataloaders)
        response = self._selector.register_training(req)
        return response.training_id

    def trainer_available(self, request: TrainerAvailableRequest, context: grpc.ServicerContext) -> TrainerAvailableResponse:
        return TrainerAvailableResponse(available=True)

    def start_training(self, request: TrainerServerRequest, context: grpc.ServicerContext) -> TrainerServerResponse:

        training_id = self.register_with_selector(request.data_info.num_dataloaders)
        print(training_id)

        # TODO(fotstrt): generalize
        train_dataloader, val_dataloader = prepare_dataloaders(
            training_id,
            request.data_info.dataset_id,
            request.data_info.num_dataloaders,
            request.batch_size
        )

        optimizer_dict = process_complex_messages(request.optimizer_parameters)
        model_conf_dict = process_complex_messages(request.model_configuration)

        model = get_model(request, optimizer_dict, model_conf_dict, train_dataloader, val_dataloader, 0)

        p = mp.Process(target=model.train)
        p.start()
        p.join()

        return TrainerServerResponse(training_id=training_id)

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
