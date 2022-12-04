from abc import ABC, abstractmethod
from data.mnist_dataset import get_mnist_dataset
import torch
from torch.optim import lr_scheduler, Adam
import pdb

from models.small_conv import SmallConv
from data.online_mnistdataset import OnlineMNISTDataset

import grpc

from dynamicdatasets.selector.selector_pb2_grpc import SelectorStub
from dynamicdatasets.selector.selector_pb2 import RegisterTrainingRequest


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
        req = RegisterTrainingRequest(training_set_size=batch_size, num_workers=num_workers)
        selector_response = self.__selector_stub.register_training(req)
        return selector_response.training_id

    def _setup_model(self):
        # TODO: Read from config
        self._model = SmallConv(self._config['trainer']['model_config'])#.to(device)

    def _scheduler_factory(self, optimizer):
        return lr_scheduler.CosineAnnealingLR(optimizer, 32)

    @abstractmethod
    def _train():
        raise NotImplementedError

    def train(self):
        self._num_epochs = self._config['trainer']['epochs']
        self._criterion = torch.nn.CrossEntropyLoss()
        self._optimizer = Adam(self._model.parameters(), lr=self._config['trainer']['lr'])
        self._scheduler = self._scheduler_factory(self._optimizer)

        # Training from selector
        training_id = self._register_training(self._config)
        print("Registered training with training id - " + str(training_id))
        train_dataset = OnlineMNISTDataset(training_id, self._config)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = self._config['trainer']['batch_size'],
                                                        num_workers=(self._config['trainer']['num_dataloader_workers']),
                                                        persistent_workers=True, shuffle = False)

        # Validation however will remain from MNIST
        val_dataset = get_mnist_dataset()['test']
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = self._config['trainer']['batch_size'], shuffle = False)
        
        self._dataloaders = {
            'train' : train_dataloader,
            'val': val_dataloader
        }

        result = self._train()

