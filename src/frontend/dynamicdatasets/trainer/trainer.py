import grpc
from models.small_conv import SmallConv
from torch.optim import lr_scheduler, Adam
import torch
from abc import ABC, abstractmethod
import os
import sys
from pathlib import Path
import logging

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from frontend.dynamicdatasets.trainer.data.mnist_dataset import get_mnist_dataset  # noqa: E402
from frontend.dynamicdatasets.trainer.data.online_mnistdataset import OnlineMNISTDataset  # noqa: E402
from backend.selector.selector_pb2_grpc import SelectorStub  # noqa: E402
from backend.selector.selector_pb2 import RegisterTrainingRequest  # noqa: E402


logging.basicConfig(format='%(asctime)s %(message)s')


class Trainer(ABC):
    _config = None

    def __init__(self, config: dict):
        self._config = config
        self._setup_model()
        self._device = 'cpu'
        self._setup_selector_stub(config)

    def _setup_selector_stub(self, config):
        selector_channel = grpc.insecure_channel(
            config['selector']['hostname'] +
            ':' +
            config['selector']['port'])
        self.__selector_stub = SelectorStub(selector_channel)

    def _register_training(self, config):
        batch_size = config['trainer']['train_set_size']
        num_workers = config['trainer']['num_dataloader_workers']
        req = RegisterTrainingRequest(
            training_set_size=batch_size,
            num_workers=num_workers)
        selector_response = self.__selector_stub.register_training(req)
        return selector_response.training_id

    def _setup_model(self):
        self._model = SmallConv(self._config['trainer']['model_config'])

    def _scheduler_factory(self, optimizer):
        return lr_scheduler.CosineAnnealingLR(optimizer, 32)

    @abstractmethod
    def _train():
        raise NotImplementedError

    def train(self):
        self._num_epochs = self._config['trainer']['epochs']
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._config['trainer']['lr'])
        self._scheduler = self._scheduler_factory(self._optimizer)

        # Training from selector
        training_id = self._register_training(self._config)
        logging.info(
            "Registered training with training id - " +
            str(training_id))
        train_dataset = OnlineMNISTDataset(training_id, self._config)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._config['trainer']['batch_size'],
            num_workers=(
                self._config['trainer']['num_dataloader_workers']),
            persistent_workers=True,
            shuffle=False)

        # Validation however will remain from MNIST
        val_dataset = get_mnist_dataset()['test']
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self._config['trainer']['batch_size'], shuffle=False)

        self._dataloaders = {
            'train': train_dataloader,
            'val': val_dataloader
        }
