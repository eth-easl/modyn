import io
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import traceback
from inspect import isfunction
from typing import Any, Callable, Optional, Union

import torch
from modyn.trainer_server.internal.dataset.data_utils import prepare_dataloaders
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.loss_callback import LossCallback
from modyn.trainer_server.internal.utils.metric_type import MetricType
from modyn.trainer_server.internal.utils.selection_strategy import get_selection_strategy
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo
from modyn.utils import dynamic_module_import, package_available_and_can_be_imported
from modyn.backend.selector.internal.selector_strategies.remote_loss_downsample import RemoteLossDownsampler

class PytorchTrainer:
    # pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches, too-many-statements

    def __init__(
        self,
        training_info: TrainingInfo,
        device: str,
        status_query_queue: mp.Queue,
        status_response_queue: mp.Queue,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.pipeline_id = training_info.pipeline_id
        self.training_id = training_info.training_id

        self._info("Initializing Pytorch Trainer")

        # setup model and optimizer
        self._model = training_info.model_handler(training_info.model_configuration_dict, device, training_info.amp)
        self._setup_optimizers(training_info)
        self._info("Model and optimizer created.")

        self._setup_lr_scheduler(training_info)
        self._info("LR scheduler created.")

        self._scaler = torch.cuda.amp.GradScaler(enabled=training_info.amp, **training_info.grad_scaler_configuration)
        self._info("Grad scaler created.")

        if training_info.use_pretrained_model:
            self._info("Loading model state from pretrained model.")
            self.load_state_if_given(training_info.pretrained_model_path, training_info.load_optimizer_state)

        criterion_func = getattr(torch.nn, training_info.torch_criterion)
        self._criterion = criterion_func(**training_info.criterion_dict)

        # setup dataloaders
        self._info("Setting up data loaders.")
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
            training_info.training_id,
        )

        self._mod_dict: dict[str, Any] = {}
        self._label_tranformer_function: Optional[Callable] = None
        if training_info.label_transformer != "":
            exec(training_info.label_transformer, self._mod_dict)  # pylint: disable=exec-used
            if "label_transformer_function" not in self._mod_dict or not isfunction(
                self._mod_dict["label_transformer_function"]
            ):
                raise ValueError("Invalid label_transformer_function is provided")
            self._label_tranformer_function = self._mod_dict["label_transformer_function"]

        self._device = device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self._amp = training_info.amp
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

        # investigate Torch RPC to make it smoother
        if training_info.selector_address != "":
            self._downsampling_enabled, exec_str = get_selection_strategy(
                training_info.selector_address, training_info.pipeline_id
            )
        else:
            self._downsampling_enabled = False

        if self._downsampling_enabled:
            exec(f"tmp = {exec_str}")  # pylint: disable=exec-used
        else:
            tmp = None

        self._downsampler = tmp

        # create callbacks - For now, assume LossCallback by default
        # TODO(#140): should be defined by the pipeline and passed with training request
        self._callbacks = {
            MetricType.LOSS: LossCallback(self._metadata_collector, criterion_func, training_info.criterion_dict)
        }

    def _setup_optimizers(self, training_info: TrainingInfo) -> None:
        self._optimizers = {}
        for name, optimizer_config in training_info.torch_optimizers_configuration.items():
            if optimizer_config["source"] == "PyTorch":
                optimizer_func = getattr(torch.optim, optimizer_config["algorithm"])
            elif optimizer_config["source"] == "APEX":
                if package_available_and_can_be_imported("apex"):
                    import apex  # pylint: disable=import-outside-toplevel, import-error

                    optimizer_func = getattr(apex.optimizers, optimizer_config["algorithm"])
                else:
                    raise ValueError("Apex Optimizer defined, but apex is not available in the system")
            else:
                raise ValueError(
                    f"Unsupported optimizer from {optimizer_config['source']}. PyTorch and APEX are supported"
                )
            optimizer_config_list = []
            for param_group in optimizer_config["param_groups"]:
                module = param_group["module"]
                param_group["config"]["params"] = eval(  # pylint: disable=eval-used
                    f"self._model.{module}.parameters()"
                )
                optimizer_config_list.append(param_group["config"])
            self._optimizers[name] = optimizer_func(optimizer_config_list)

    def _setup_lr_scheduler(self, training_info: TrainingInfo) -> None:
        self._lr_scheduler = None
        if training_info.lr_scheduler:
            if training_info.lr_scheduler["source"] == "Custom":
                lr_scheduler_module = dynamic_module_import("modyn.trainer_server.custom_lr_schedulers")
                custom_lr_scheduler = getattr(lr_scheduler_module, training_info.lr_scheduler["name"])
                optimizers = [self._optimizers[opt] for opt in training_info.lr_scheduler["optimizers"]]
                self._lr_scheduler = custom_lr_scheduler(optimizers, training_info.lr_scheduler["config"])
            elif training_info.lr_scheduler["source"] == "PyTorch":
                torch_lr_scheduler = getattr(torch.optim.lr_scheduler, training_info.lr_scheduler["name"])
                if len(training_info.lr_scheduler["optimizers"]) > 1:
                    self._warning("Provided a LR scheduler from PyTorch, but multiple optimizers")
                self._lr_scheduler = torch_lr_scheduler(
                    self._optimizers[training_info.lr_scheduler["optimizers"][0]],
                    **training_info.lr_scheduler["config"],
                )
            else:
                raise ValueError(
                    f"Unsupported LR scheduler of source {training_info.lr_scheduler['source']}."
                    "PyTorch and Custom are supported"
                )

    def _info(self, msg: str) -> None:
        self.logger.info(f"[Training {self.training_id}][PL {self.pipeline_id}] {msg}")

    def _warning(self, msg: str) -> None:
        self.logger.warning(f"[Training {self.training_id}][PL {self.pipeline_id}] {msg}")

    def save_state(self, destination: Union[pathlib.Path, io.BytesIO], iteration: Optional[int] = None) -> None:
        dict_to_save = {}
        dict_to_save["model"] = self._model.model.state_dict()
        for optimizer_name, optimizer in self._optimizers.items():
            dict_to_save[f"optimizer-{optimizer_name}"] = optimizer.state_dict()

        if iteration is not None:
            dict_to_save["iteration"] = iteration

        torch.save(dict_to_save, destination)

    def load_state_if_given(self, path: pathlib.Path, load_optimizer_state: bool = False) -> None:
        assert path.exists(), "Cannot load state from non-existing file"
        self._info(f"Loading model state from {path}")
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()))

        assert "model" in checkpoint
        self._model.model.load_state_dict(checkpoint["model"])
        if load_optimizer_state:
            for optimizer_name, optimizer in self._optimizers.items():
                if f"optimizer-{optimizer_name}" in checkpoint:
                    optimizer.load_state_dict(checkpoint[f"optimizer-{optimizer_name}"])

        os.remove(path)

    def send_model_state_to_server(self) -> None:
        buffer = io.BytesIO()
        self.save_state(buffer)
        buffer.seek(0)
        bytes_state = buffer.read()
        self._status_response_queue.put(bytes_state)

    def send_status_to_server(self, batch_number: int) -> None:
        self._status_response_queue.put({"num_batches": batch_number, "num_samples": self._num_samples})

    def train(self) -> None:  # pylint: disable=too-many-locals, too-many-branches
        self._info(f"Process {os.getpid()} starts training")

        self._model.model.train()

        for _, callback in self._callbacks.items():
            callback.on_train_begin(self._model.model, self._optimizers)

        self._info("Handled OnBegin Callbacks.")

        batch_number = 0
        for batch_number, batch in enumerate(self._train_dataloader):
            for _, callback in self._callbacks.items():
                callback.on_batch_begin(self._model.model, self._optimizers, batch, batch_number)

            # As empty() is unreliable
            # we try to fetch an element within 10ms. If there is no
            # element within that timeframe returned, we continue.
            try:
                req = self._status_query_queue.get(timeout=0.01)
                if req == TrainerMessages.STATUS_QUERY_MESSAGE:
                    self.send_status_to_server(batch_number)
                elif req == TrainerMessages.MODEL_STATE_QUERY_MESSAGE:
                    self.send_model_state_to_server()
                else:
                    raise ValueError("Unknown message in the status query queue")
            except queue.Empty:
                pass

            sample_ids = batch[0]
            if isinstance(sample_ids, torch.Tensor):
                sample_ids = sample_ids.tolist()
            elif isinstance(sample_ids, tuple):
                sample_ids = list(sample_ids)

            assert isinstance(sample_ids, list), "Cannot parse result from DataLoader"

            if self._label_tranformer_function is None:
                target = batch[2].to(self._device)
            else:
                target = self._label_tranformer_function(batch[2]).to(self._device)

            data: Union[torch.Tensor, dict]
            if isinstance(batch[1], torch.Tensor):
                data = batch[1].to(self._device)
            elif isinstance(batch[1], dict):
                data: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                for name, tensor in batch[1].items():
                    data[name] = tensor.to(self._device)

            if self._downsampling_enabled:
                # need to refactor to allow weights
                assert self._downsampler is not None
                data, _, target = self._downsampler.sample(data, target)

            for _, optimizer in self._optimizers.items():
                optimizer.zero_grad()

            # TODO(#163): where to perform lr_scheduler.step? make it configurable
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

            with torch.autocast(self._device_type, enabled=self._amp):
                output = self._model.model(data)
                loss = self._criterion(output, target)

            self._scaler.scale(loss).backward()

            for _, callback in self._callbacks.items():
                callback.on_batch_before_update(
                    self._model.model, self._optimizers, batch_number, sample_ids, data, target, output, loss
                )

            for _, optimizer in self._optimizers.items():
                self._scaler.step(optimizer)

            self._scaler.update()

            if self._checkpoint_interval > 0 and batch_number % self._checkpoint_interval == 0:
                checkpoint_file_name = self._checkpoint_path / f"model_{batch_number}.modyn"
                self.save_state(checkpoint_file_name, batch_number)

            self._num_samples += target.shape[0]

            for _, callback in self._callbacks.items():
                callback.on_batch_end(
                    self._model.model, self._optimizers, batch_number, sample_ids, data, target, output, loss
                )

        self._info(f"Finished training: {self._num_samples} samples, {batch_number} batches.")
        for _, callback in self._callbacks.items():
            callback.on_train_end(self._model.model, self._optimizers, self._num_samples, batch_number)

        for metric in self._callbacks:
            self._metadata_collector.send_metadata(metric)
        self._metadata_collector.cleanup()

        # save final model
        final_checkpoint_file_name = self._final_checkpoint_path / "model_final.modyn"
        self.save_state(final_checkpoint_file_name)

        self._info("Training complete!")


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
