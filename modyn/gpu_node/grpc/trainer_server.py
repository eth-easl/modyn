import grpc
import os
import sys
from concurrent import futures
from pathlib import Path
import logging
import torch
import multiprocessing as mp

import yaml

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modyn.gpu_node.grpc.trainer_server_pb2_grpc import TrainerServerServicer, add_TrainerServerServicer_to_server
from modyn.gpu_node.grpc.trainer_server_pb2 import TrainerServerRequest, TrainerServerResponse

# TODO(fotstrt): replace with dynamic loading
from modyn.gpu_node.data.cifar_dataset import get_cifar_datasets
from modyn.gpu_node.models.resnet import ResNet

logging.basicConfig(format='%(asctime)s %(message)s')

from modyn.backend.selector.mock_selector_server import MockSelectorServer, RegisterTrainingRequest
from modyn.gpu_node.data.online_dataset import OnlineDataset

class TrainerGRPCServer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self):
        self._selector = MockSelectorServer()

    def prepare_dataloaders(self, training_id, dataset_id, num_dataloaders, batch_size):

        if dataset_id == "cifar10":
            train_set, val_set = get_cifar_datasets()

            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                            shuffle=True, num_workers=num_dataloaders)

            val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                            shuffle=False, num_workers=num_dataloaders)

        elif dataset_id == "online":
            train_set = OnlineDataset(training_id, None)
            # TODO(fotstrt): what to do with the val set?

            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                            num_workers=num_dataloaders)

            val_dataloader = None

        else:
            return None, None

        return train_dataloader, val_dataloader


    def register_with_selector(self, num_dataloaders):
        # TODO: replace this with grpc calls to the selector
        req = RegisterTrainingRequest(num_workers=num_dataloaders)
        response = self._selector.register_training(req)
        return response.training_id

    def start_training(self, request: TrainerServerRequest, context: grpc.ServicerContext) -> TrainerServerResponse:

        training_id = self.register_with_selector(request.data_info.num_dataloaders)
        print(training_id)

        # TODO(fotstrt): generalize
        train_dataloader, val_dataloader = self.prepare_dataloaders(
            training_id,
            request.data_info.dataset_id,
            request.data_info.num_dataloaders,
            request.hyperparameters.batch_size
        )


        # this is tailor-made for a specific example
        # TODO(fotstrt): generalize
        model = ResNet(
            'resnet18',
            'SGD',
            {'lr': 0.1},
            torch.nn.CrossEntropyLoss,
            10,
            train_dataloader,
            val_dataloader,
            0
        )

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
