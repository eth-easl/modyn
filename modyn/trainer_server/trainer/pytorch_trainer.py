import io
import traceback
from typing import Optional
import torch
import logging
import os
import multiprocessing as mp


from modyn.trainer_server.dataset.utils import prepare_dataloaders
from modyn.trainer_server.utils.model_utils import get_model
from modyn.trainer_server.utils.training_utils import STATUS_QUERY_MESSAGE, TrainingInfo


class PytorchTrainer:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        training_info: TrainingInfo,
        device: str,
        train_until_sample_id: str,
        status_query_queue: mp.Queue,
        status_response_queue: mp.Queue,
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
            training_info.transform_list,
            train_until_sample_id,
        )

        self._device = device
        self._checkpoint_path = training_info.checkpoint_path
        self._checkpoint_interval = training_info.checkpoint_interval
        self._logger = logging.getLogger()

        if not os.path.isdir(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)

        self._status_query_queue = status_query_queue
        self._status_response_queue = status_response_queue

    def create_logger(self, log_path: str) -> None:

        self._logger = logging.getLogger('trainer')
        self._logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s – %(message)s')
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.propagate = False

    def save_checkpoint(self, checkpoint_file_name: str, iteration: int) -> None:

        dict_to_save = {
            'model': self._model.model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'iteration': iteration,
        }
        torch.save(dict_to_save, checkpoint_file_name)

    def load_checkpoint(self, path: str) -> None:

        checkpoint_dict = torch.load(path)

        assert 'model' in checkpoint_dict
        assert 'optimizer' in checkpoint_dict

        self._model.model.load_state_dict(checkpoint_dict['model'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])

    def send_state_to_server(self, iteration: int) -> None:

        dict_to_send = {
            'model': self._model.model.state_dict(),
            'optimizer': self._optimizer.state_dict(),
        }

        buffer = io.BytesIO()
        torch.save(dict_to_send, buffer)
        buffer.seek(0)
        bytes_state = buffer.read()
        self._status_response_queue.put({'state': bytes_state, 'iteration': iteration})

    def train(self, log_path: str, load_checkpoint_path: Optional[str] = None) -> None:

        self.create_logger(log_path)

        self._logger.info(f'Process {os.getpid()} starts training')

        if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
            self.load_checkpoint(load_checkpoint_path)

        self._model.model.train()

        train_iter = enumerate(self._train_dataloader)

        for i, batch in train_iter:

            if not self._status_query_queue.empty():
                req = self._status_query_queue.get()
                assert req == STATUS_QUERY_MESSAGE
                self.send_state_to_server(i)

            self._optimizer.zero_grad()
            data, target = batch[0].to(self._device), batch[1].to(self._device)
            output = self._model.model(data)
            loss = self._criterion(output, target)
            loss.backward()
            self._optimizer.step()

            if self._checkpoint_interval > 0 and i % self._checkpoint_interval == 0:
                checkpoint_file_name = self._checkpoint_path + f'/model_{i}' + '.pt'
                self.save_checkpoint(checkpoint_file_name, i)

            self._logger.info(f'Iteration {i}')

        self._logger.info('Training complete!')


def train(
    training_info: TrainingInfo,
    device: str,
    log_path: str,
    load_checkpoint_path: Optional[str],
    train_until_sample_id: str,
    exception_queue: mp.Queue,
    status_query_queue: mp.Queue,
    status_response_queue: mp.Queue
) -> None:

    try:
        trainer = PytorchTrainer(training_info, device, train_until_sample_id, status_query_queue, status_response_queue)
        trainer.train(log_path, load_checkpoint_path)
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        exception_queue.put(exception_msg)
