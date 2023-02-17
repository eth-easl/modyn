import io
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import traceback
from typing import Optional, Union

import torch
from modyn.trainer_server.internal.dataset.data_utils import prepare_dataloaders
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.loss_callback import LossCallback
from modyn.trainer_server.internal.utils.metric_type import MetricType
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo


class PytorchTrainer:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        training_info: TrainingInfo,
        device: str,
        status_query_queue: mp.Queue,
        status_response_queue: mp.Queue,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.logger.info("Initializing Pytorch Trainer")

        # setup model and optimizer
        self._model = training_info.model_handler(training_info.model_configuration_dict)
        self._model.model.to(device)

        optimizer_func = getattr(torch.optim, training_info.torch_optimizer)
        self._optimizer = optimizer_func(self._model.model.parameters(), **training_info.optimizer_dict)

        self.logger.info("Model and optimizer created.")

        if training_info.used_pretrained_model:
            self.logger.info("Loading model state from pretrained model.")
            self.load_state_if_given(training_info.pretrained_model, training_info.load_optimizer_state)

        criterion_func = getattr(torch.nn, training_info.torch_criterion)
        self._criterion = criterion_func(**training_info.criterion_dict)

        # setup dataloaders
        self.logger.info("Setting up data loaders.")
        self._train_dataloader, self._val_dataloader = prepare_dataloaders(
            training_info.pipeline_id,
            training_info.trigger_id,
            training_info.dataset_id,
            training_info.num_dataloaders,
            training_info.batch_size,
            training_info.bytes_parser,
            training_info.transform_list,
            training_info.storage_address,
            training_info.selector_address,
        )

        self._device = device
        self._checkpoint_path = training_info.checkpoint_path
        self._checkpoint_interval = training_info.checkpoint_interval
        self._final_checkpoint_path = training_info.final_checkpoint_path

        if not self._checkpoint_path.is_dir():
            self._checkpoint_path.mkdir()

        self._final_checkpoint_path.mkdir()  # exist_ok == False, this directory should not exist before

        self._status_query_queue = status_query_queue
        self._status_response_queue = status_response_queue

        self._num_samples = 0

        self._metadata_collector = MetadataCollector(training_info.pipeline_id, training_info.trigger_id)

        # create callbacks - For now, assume LossCallback by default
        # TODO(#140): should be defined by the pipeline and passed with training request
        self._callbacks = {
            MetricType.LOSS: LossCallback(self._metadata_collector, criterion_func, training_info.criterion_dict)
        }

    def save_state(self, destination: Union[pathlib.Path, io.BytesIO], iteration: Optional[int] = None) -> None:
        dict_to_save = {
            "model": self._model.model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }
        if iteration is not None:
            dict_to_save["iteration"] = iteration

        torch.save(dict_to_save, destination)

    def load_state_if_given(self, initial_state: bytes, load_optimizer_state: bool = False) -> None:
        checkpoint_buffer = io.BytesIO(initial_state)
        checkpoint = torch.load(checkpoint_buffer)
        assert "model" in checkpoint
        self._model.model.load_state_dict(checkpoint["model"])
        if load_optimizer_state and "optimizer" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer"])

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

    def train(self) -> None:
        self.logger.info(f"Process {os.getpid()} starts training")

        self._model.model.train()

        for _, callback in self._callbacks.items():
            callback.on_train_begin(self._model.model, self._optimizer)

        batch_number = 0
        for batch_number, batch in enumerate(self._train_dataloader):
            for _, callback in self._callbacks.items():
                callback.on_batch_begin(self._model.model, self._optimizer, batch, batch_number)

            # As empty() is unreliable
            # we try to fetch an element within 10ms. If there is no
            # element within that timeframe returned, we continue.
            try:
                req = self._status_query_queue.get(timeout=0.01)
                if req != TrainerMessages.STATUS_QUERY_MESSAGE:
                    raise ValueError("Unknown message in the status query queue")
                self.send_state_to_server(batch_number)
            except queue.Empty:
                pass

            sample_ids = batch[0]
            data, target = batch[1].to(self._device), batch[2].to(self._device)

            self._optimizer.zero_grad()
            output = self._model.model(data)
            loss = self._criterion(output, target)
            loss.backward()

            for _, callback in self._callbacks.items():
                callback.on_batch_before_update(
                    self._model.model, self._optimizer, batch_number, sample_ids, data, target, output, loss
                )

            self._optimizer.step()

            if self._checkpoint_interval > 0 and batch_number % self._checkpoint_interval == 0:
                checkpoint_file_name = self._checkpoint_path / f"model_{batch_number}.modyn"
                self.save_state(checkpoint_file_name, batch_number)

            self._num_samples += data.shape[0]

            for _, callback in self._callbacks.items():
                callback.on_batch_end(
                    self._model.model, self._optimizer, batch_number, sample_ids, data, target, output, loss
                )

        self.logger.info(f"Finished training: {self._num_samples} samples, {batch_number} batches.")
        for _, callback in self._callbacks.items():
            callback.on_train_end(self._model.model, self._optimizer, self._num_samples, batch_number)

        for metric in self._callbacks:
            self._metadata_collector.send_metadata(metric)
        self._metadata_collector.cleanup()

        # save final model
        final_checkpoint_file_name = self._final_checkpoint_path / "model_final.modyn"
        self.save_state(final_checkpoint_file_name)

        self.logger.info("Training complete!")


def train(
    training_info: TrainingInfo,
    device: str,
    log_path: pathlib.Path,
    exception_queue: mp.Queue,
    status_query_queue: mp.Queue,
    status_response_queue: mp.Queue,
) -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s]  [%(filename)15s:%(lineno)4d] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)

    try:
        trainer = PytorchTrainer(training_info, device, status_query_queue, status_response_queue, logger)
        trainer.train()
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)
