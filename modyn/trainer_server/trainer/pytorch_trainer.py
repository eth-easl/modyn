import traceback
from typing import Optional
import torch
import logging
import os
import time

from modyn.trainer_server.dataset.utils import prepare_dataloaders
from modyn.trainer_server.utils.model_utils import get_model


class PytorchTrainer:
    def __init__(
        self,
        training_info,
        device,
    ) -> None:

        # setup model and optimizer
        self._model = get_model(training_info.model_id, training_info.model_configuration_dict)
        self._model.model.to(device)

        optimizer_func = getattr(torch.optim, training_info.torch_optimizer)
        self._optimizer = optimizer_func(self._model.model.parameters(), **training_info.optimizer_dict)

        criterion_func = getattr(torch.nn, training_info.torch_criterion)
        self._criterion = criterion_func(**training_info.criterion_dict)

        # setup dataloaders
        self._train_dataloader, self._val_dataloader = prepare_dataloaders(
            training_info.training_id,
            training_info.dataset_id,
            training_info.num_dataloaders,
            training_info.batch_size,
            training_info.transform_list
        )

        # setup rest
        self._device = device
        self._checkpoint_path = training_info.checkpoint_path
        self._checkpoint_interval = training_info.checkpoint_interval

        if not os.path.isdir(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)

    def create_logger(self, log_path: str):

        self._logger = logging.getLogger('test')  # TODO(fotstrt): fix this
        self._logger.setLevel(logging.INFO)

        fileHandler = logging.FileHandler(log_path)
        fileHandler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s – %(message)s')
        fileHandler.setFormatter(formatter)

        self._logger.addHandler(fileHandler)
        self._logger.propagate = False

    def save_checkpoint(self, checkpoint_file_name: str):

        # TODO(fotstrt): this might overwrite checkpoints from previous runs
        # we could have a counter for the specific training, and increment it
        # every time a new checkpoint is saved.

        # TODO: we assume a local checkpoint for now,
        # should we add functionality for remote?

        dict_to_save = {
            'model': self._model.model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }
        torch.save(dict_to_save, checkpoint_file_name)

    def load_checkpoint(self, path: str):

        checkpoint_dict = torch.load(path)

        assert 'model' in checkpoint_dict
        assert 'optimizer' in checkpoint_dict

        self._model.model.load_state_dict(checkpoint_dict['model'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])

    def train(self, log_path: str, load_checkpoint_path=Optional[str]):

        self.create_logger(log_path)

        self._logger.info('Process {} starts training'.format(os.getpid()))

        if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
            self.load_checkpoint(load_checkpoint_path)

        self._model.model.train()

        train_iter = enumerate(self._train_dataloader)

        for i, batch in train_iter:

            self._optimizer.zero_grad()
            data, target = batch[0].to(self._device), batch[1].to(self._device)
            output = self._model.model(data)
            loss = self._criterion(output, target)
            loss.backward()
            self._optimizer.step()

            if self._checkpoint_interval > 0 and i % self._checkpoint_interval == 0:
                checkpoint_file_name = self._checkpoint_path + f'/model_{i}' + '.pt'
                self.save_checkpoint(checkpoint_file_name)

            self._logger.info('Iteration {}'.format(i))

        self._logger.info('Training complete!')


def train(training_info, device, log_path, load_checkpoint_path):

    trainer = PytorchTrainer(training_info, device)
    trainer.train(log_path, load_checkpoint_path)
