import io
import logging
import multiprocessing as mp
import os
import traceback
from typing import Optional, Union

import torch
from modyn.trainer_server.internal.dataset.data_utils import prepare_dataloaders
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo

logger = logging.getLogger(__name__)


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
        self._model = training_info.model_handler(training_info.model_configuration_dict)
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
            training_info.bytes_parser,
            training_info.transform_list,
            train_until_sample_id,
        )

        self._device = device
        self._checkpoint_path = training_info.checkpoint_path
        self._checkpoint_interval = training_info.checkpoint_interval

        if not os.path.isdir(self._checkpoint_path):
            os.mkdir(self._checkpoint_path)

        self._status_query_queue = status_query_queue
        self._status_response_queue = status_response_queue

        self._num_samples = 0

    def save_state(self, destination: Union[str, io.BytesIO], iteration: Optional[int] = None) -> None:

        dict_to_save = {
            "model": self._model.model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        if iteration is not None:
            dict_to_save["iteration"] = iteration

        torch.save(dict_to_save, destination)

    def load_checkpoint(self, path: str) -> None:

        checkpoint_dict = torch.load(path)

        assert "model" in checkpoint_dict
        assert "optimizer" in checkpoint_dict

        self._model.model.load_state_dict(checkpoint_dict["model"])
        self._optimizer.load_state_dict(checkpoint_dict["optimizer"])

    def send_state_to_server(self, batch_number: int) -> None:

        buffer = io.BytesIO()
        self.save_state(buffer)
        buffer.seek(0)
        bytes_state = buffer.read()
        self._status_response_queue.put(
            {
                "state": bytes_state,
                "num_batches": batch_number,
                "num_samples": self._num_samples,
            }
        )

    def train(self, log_path: str, load_checkpoint_path: Optional[str] = None) -> None:

        file_handler = logging.FileHandler(log_path)
        logger.addHandler(file_handler)

        logger.info(f"Process {os.getpid()} starts training")

        if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
            self.load_checkpoint(load_checkpoint_path)

        self._model.model.train()

        train_iter = enumerate(self._train_dataloader)

        for batch_number, batch in train_iter:

            if not self._status_query_queue.empty():
                req = self._status_query_queue.get()
                if req != TrainerMessages.STATUS_QUERY_MESSAGE:
                    raise ValueError("Unknown message in the status query queue")
                self.send_state_to_server(batch_number)

            self._optimizer.zero_grad()
            data, target = batch[0].to(self._device), batch[1].to(self._device)
            output = self._model.model(data)
            loss = self._criterion(output, target)
            loss.backward()
            self._optimizer.step()

            if self._checkpoint_interval > 0 and batch_number % self._checkpoint_interval == 0:
                checkpoint_file_name = self._checkpoint_path + f"/model_{batch_number}" + ".pt"
                self.save_state(checkpoint_file_name, batch_number)

            self._num_samples += batch[0].shape[0]

            logger.info(f"Iteration {batch_number}")

        logger.info("Training complete!")
        logger.removeHandler(file_handler)


def train(
    training_info: TrainingInfo,
    device: str,
    log_path: str,
    load_checkpoint_path: Optional[str],
    train_until_sample_id: str,
    exception_queue: mp.Queue,
    status_query_queue: mp.Queue,
    status_response_queue: mp.Queue,
) -> None:

    try:
        trainer = PytorchTrainer(
            training_info,
            device,
            train_until_sample_id,
            status_query_queue,
            status_response_queue,
        )
        trainer.train(log_path, load_checkpoint_path)
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)
