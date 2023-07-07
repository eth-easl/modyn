# pylint: disable=no-name-in-module
import io
import json
import logging
import multiprocessing as mp
import os
import pathlib
import queue
import traceback
from enum import Enum
from typing import Optional, Tuple, Union

import grpc
import torch
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    AvailableLabelsResponse,
    GetAvailableLabelsRequest,
    GetSelectionStrategyRequest,
    SelectionStrategyResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.trainer_server.internal.dataset.data_utils import (
    prepare_dataloaders,
    prepare_per_class_dataloader_from_online_dataset,
)
from modyn.trainer_server.internal.dataset.key_sources import LocalKeySource, SelectorKeySource
from modyn.trainer_server.internal.dataset.local_dataset_writer import LocalDatasetWriter
from modyn.trainer_server.internal.metadata_collector.metadata_collector import MetadataCollector
from modyn.trainer_server.internal.trainer.metadata_pytorch_callbacks.loss_callback import LossCallback
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsample_strategy import (
    AbstractRemoteDownsamplingStrategy,
    get_tensors_subset,
)
from modyn.trainer_server.internal.utils.metric_type import MetricType
from modyn.trainer_server.internal.utils.trainer_messages import TrainerMessages
from modyn.trainer_server.internal.utils.training_info import TrainingInfo
from modyn.utils import (
    LABEL_TRANSFORMER_FUNC_NAME,
    DownsamplingMode,
    deserialize_function,
    dynamic_module_import,
    grpc_connection_established,
    package_available_and_can_be_imported,
    seed_everything,
)

AvailableQueues = Enum("AvailableQueues", ["TRAINING", "DOWNSAMPLING"])


class PytorchTrainer:
    # pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches, too-many-statements

    def __init__(
        self,
        training_info: TrainingInfo,
        device: str,
        status_query_queue_training: mp.Queue,
        status_response_queue_training: mp.Queue,
        status_query_queue_downsampling: mp.Queue,
        status_response_queue_downsampling: mp.Queue,
        logger: logging.Logger,
    ) -> None:
        self.logger = logger
        self.pipeline_id = training_info.pipeline_id
        self.training_id = training_info.training_id
        self.trigger_id = training_info.trigger_id

        self.selector_stub = self.connect_to_selector(training_info.selector_address)

        if training_info.seed is not None:
            self.seed_trainer_server(training_info.seed)
            self._info("Everything seeded")

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

        self._batch_size = training_info.batch_size
        self._num_dataloaders = training_info.num_dataloaders

        self._label_tranformer_function = deserialize_function(
            training_info.label_transformer, LABEL_TRANSFORMER_FUNC_NAME
        )

        self._device = device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self._amp = training_info.amp
        self._checkpoint_path = training_info.checkpoint_path
        self._checkpoint_interval = training_info.checkpoint_interval
        self._final_checkpoint_path = training_info.final_checkpoint_path
        self.epochs_per_trigger = training_info.epochs_per_trigger

        if not self._checkpoint_path.is_dir():
            self._checkpoint_path.mkdir()

        self._final_checkpoint_path.mkdir()  # exist_ok == False, this directory should not exist before

        self._status_query_queue_training = status_query_queue_training
        self._status_response_queue_training = status_response_queue_training

        self._status_query_queue_downsampling = status_query_queue_downsampling
        self._status_response_queue_downsampling = status_response_queue_downsampling

        self._num_samples = 0

        self._metadata_collector = MetadataCollector(self.pipeline_id, self.trigger_id)

        self.selector_stub = self.connect_to_selector(training_info.selector_address)
        self.selector_address = training_info.selector_address

        downsampling_enabled, strategy_name, downsampler_config = self.get_selection_strategy()
        if downsampling_enabled:
            self._setup_downsampling(criterion_func, downsampler_config, strategy_name, training_info)
        else:
            self._downsampling_mode = DownsamplingMode.DISABLED

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

        # create callbacks - For now, assume LossCallback by default
        # TODO(#140): should be defined by the pipeline and passed with training request
        self._callbacks = {
            MetricType.LOSS: LossCallback(self._metadata_collector, criterion_func, training_info.criterion_dict)
        }

    def _setup_downsampling(
        self,
        criterion_func: torch.nn.modules.loss,
        downsampler_config: dict,
        strategy_name: str,
        training_info: TrainingInfo,
    ) -> None:
        self._criterion_nored = criterion_func(**training_info.criterion_dict, reduction="none")
        self._downsampler = self.instantiate_downsampler(strategy_name, downsampler_config, self._criterion_nored)
        assert "sample_then_batch" in downsampler_config
        if downsampler_config["sample_then_batch"]:
            self._downsampling_mode = DownsamplingMode.SAMPLE_THEN_BATCH
            assert "downsampling_period" in downsampler_config
            self._downsampling_period = downsampler_config["downsampling_period"]
            self.offline_dataset_path = training_info.offline_dataset_path
        else:
            self._downsampling_mode = DownsamplingMode.BATCH_THEN_SAMPLE

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
        self._status_response_queue_training.put(bytes_state)

    def send_status_to_server_training(self, batch_number: int) -> None:
        self._status_response_queue_training.put({"num_batches": batch_number, "num_samples": self._num_samples})

    def get_selection_strategy(self) -> tuple[bool, str, dict]:
        req = GetSelectionStrategyRequest(pipeline_id=self.pipeline_id)

        response: SelectionStrategyResponse = self.selector_stub.get_selection_strategy(req)
        downsampler_config = json.loads(response.downsampler_config.value)

        return response.downsampling_enabled, response.strategy_name, downsampler_config

    def seed_trainer_server(self, seed: int) -> None:
        if not (0 <= seed <= 100 and isinstance(seed, int)):
            raise ValueError("The seed must be an integer in the range [0,100]")
        # seed the trainer server
        seed_everything(seed)

    def connect_to_selector(self, selector_address: str) -> SelectorStub:
        selector_channel = grpc.insecure_channel(selector_address)
        assert selector_channel is not None
        if not grpc_connection_established(selector_channel):
            raise ConnectionError(f"Could not establish gRPC connection to selector at address {selector_address}.")
        return SelectorStub(selector_channel)

    def instantiate_downsampler(
        self, strategy_name: str, downsampler_config: dict, per_sample_loss: torch.nn.modules.loss
    ) -> AbstractRemoteDownsamplingStrategy:
        remote_downsampling_module = dynamic_module_import("modyn.trainer_server.internal.trainer.remote_downsamplers")
        downsampler_class = getattr(remote_downsampling_module, strategy_name)

        downsampler = downsampler_class(
            self.pipeline_id, self.trigger_id, self._batch_size, downsampler_config, per_sample_loss
        )
        return downsampler

    def sample_then_batch_this_epoch(self, epoch: int) -> bool:
        if self._downsampling_mode != DownsamplingMode.SAMPLE_THEN_BATCH:
            return False

        # self._downsampling_period = 0 : downsample one time per trigger
        if self._downsampling_period == 0:
            return epoch == 0
        # otherwise dowsample every self._downsampling_period epochs
        return epoch % self._downsampling_period == 0

    def train(self) -> None:  # pylint: disable=too-many-locals, too-many-branches
        self._info(f"Process {os.getpid()} starts training")

        self._model.model.train()

        for _, callback in self._callbacks.items():
            callback.on_train_begin(self._model.model, self._optimizers)

        self._info("Handled OnBegin Callbacks.")

        batch_number = -1
        for epoch in range(self.epochs_per_trigger):
            if self.sample_then_batch_this_epoch(epoch):
                self.update_queue(AvailableQueues.TRAINING, batch_number, self._num_samples, training_active=False)
                self.downsample_trigger_training_set()

            for batch_number, batch in enumerate(self._train_dataloader):
                retrieve_weights_from_dataloader, weighted_optimization = self.weights_handling(len(batch))

                for _, callback in self._callbacks.items():
                    callback.on_batch_begin(self._model.model, self._optimizers, batch, batch_number)

                self.update_queue(AvailableQueues.TRAINING, batch_number, self._num_samples, training_active=True)

                sample_ids, target, data = self.preprocess_batch(batch)

                if retrieve_weights_from_dataloader:
                    # model output is a torch.FloatTensor but weights is a torch.DoubleTensor.
                    # We need to cast to do the dot product
                    weights = batch[3].float()

                for _, optimizer in self._optimizers.items():
                    optimizer.zero_grad()

                # TODO(#163): where to perform lr_scheduler.step? make it configurable
                if self._lr_scheduler is not None:
                    self._lr_scheduler.step()

                pre_downsampling_size = target.shape[0]

                with torch.autocast(self._device_type, enabled=self._amp):
                    if self._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
                        data, sample_ids, target, weights = self.downsample_batch(data, sample_ids, target)

                    output = self._model.model(data)

                    if weighted_optimization:
                        # weighted gradient descent
                        assert weights is not None
                        loss = torch.dot(self._criterion_nored(output, target), weights / weights.sum())
                    else:
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

                self._num_samples += pre_downsampling_size

                for _, callback in self._callbacks.items():
                    callback.on_batch_end(
                        self._model.model, self._optimizers, batch_number, sample_ids, data, target, output, loss
                    )

        self._info(f"Finished training: {self._num_samples} samples, {batch_number + 1} batches.")
        for _, callback in self._callbacks.items():
            callback.on_train_end(self._model.model, self._optimizers, self._num_samples, batch_number)

        for metric in self._callbacks:
            self._metadata_collector.send_metadata(metric)
        self._metadata_collector.cleanup()

        # save final model
        final_checkpoint_file_name = self._final_checkpoint_path / "model_final.modyn"
        self.save_state(final_checkpoint_file_name)

        # clean temporary directories in dataloader
        self.end_of_trigger_cleaning()

        self._info("Training complete!")

    def weights_handling(self, batch_len: int) -> Tuple[bool, bool]:
        # whether the dataloader returned the weights.
        retrieve_weights_from_dataloader = batch_len == 4  # key, sample, label, weight

        # we want to use weighted optimization if we get weights from the dataloader or if we compute them in the
        # training loop (BATCH_THEN_SAMPLE downsampling mode)
        weighted_optimization = (
            retrieve_weights_from_dataloader or self._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE
        )

        return retrieve_weights_from_dataloader, weighted_optimization

    def update_queue(
        self, queue_name: AvailableQueues, batch_number: int, number_of_samples: int, training_active: bool
    ) -> None:
        if queue_name == AvailableQueues.TRAINING:
            queue_in = self._status_query_queue_training
            queue_out = self._status_response_queue_training
        elif queue_name == AvailableQueues.DOWNSAMPLING:
            queue_in = self._status_query_queue_downsampling
            queue_out = self._status_response_queue_downsampling
        else:
            raise AssertionError(f"Queue {queue_name} does not exist.")

        # As empty() is unreliable
        # we try to fetch an element within 10ms. If there is no
        # element within that timeframe returned, we continue.
        try:
            req = queue_in.get(timeout=0.01)
            if req == TrainerMessages.STATUS_QUERY_MESSAGE:
                queue_out.put(
                    {"num_batches": batch_number, "num_samples": number_of_samples, "training_active": training_active}
                )
            elif req == TrainerMessages.MODEL_STATE_QUERY_MESSAGE:
                self.send_model_state_to_server()
            else:
                raise ValueError("Unknown message in the status query queue")
        except queue.Empty:
            pass

    def preprocess_batch(self, batch: tuple) -> tuple[list, torch.Tensor, Union[torch.Tensor, dict]]:
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
        else:
            raise ValueError(
                "The format of the data provided is not supported in modyn. "
                "Please use either torch tensors or dict[str, torch.Tensor]"
            )

        return sample_ids, target, data

    def downsample_batch(
        self, data: torch.Tensor, sample_ids: list, target: torch.Tensor
    ) -> Tuple[torch.Tensor, list, torch.Tensor, torch.Tensor]:
        """
        Function to score every datapoint in the current BATCH and sample a fraction of it
        Used for downsampling strategies in BATCH_THEN_SAMPLE mode

        Receives the samples, the sample ids and the targets. Returns the selected subset of these
        tensors and the weights for each sample.
        """

        assert self._downsampler is not None
        assert self._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE

        self._downsampler.init_downsampler()
        big_batch_output = self._model.model(data)
        self._downsampler.inform_samples(sample_ids, big_batch_output, target)
        # TODO(#218) Persist information on the sample IDs/weights when downsampling is performed
        selected_indexes, weights = self._downsampler.select_points()
        selected_data, selected_target = get_tensors_subset(selected_indexes, data, target, sample_ids)
        sample_ids, data, target = selected_indexes, selected_data, selected_target
        # TODO(#219) Investigate if we can avoid 2 forward passes
        return data, sample_ids, target, weights

    def downsample_trigger_training_set(self) -> None:
        """
        Function to score every datapoint in the current PRESAMPLED DATASET and sample a fraction of it
        Used for downsampling strategies in SAMPLE_THEN_BATCH mode

        """
        assert self._downsampler is not None
        assert self._downsampling_mode == DownsamplingMode.SAMPLE_THEN_BATCH

        # set the model to eval to avoid errors like Expected more than 1 value per channel when training, got ...
        self._model.model.eval()
        # keys must be taken from the selector.
        # This operation is needed only when we sample several times (otherwise the source is already the selector)
        selector_key_source = SelectorKeySource(
            pipeline_id=self.pipeline_id, trigger_id=self.trigger_id, selector_address=self.selector_address
        )
        self._train_dataloader.dataset.change_key_source(selector_key_source)
        self._downsampler.init_downsampler()
        if self._downsampler.requires_data_label_by_label:
            available_labels = self._get_available_labels_from_selector()
            per_class_dataloader = prepare_per_class_dataloader_from_online_dataset(
                self._train_dataloader.dataset, self._batch_size, self._num_dataloaders
            )
            number_of_samples = 0
            batch_number = 0
            for label in available_labels:
                per_class_dataloader.dataset.filtered_label = label
                batch_number, number_of_samples = self._iterate_dataloader_and_compute_scores(
                    per_class_dataloader,
                    previous_batch_number=batch_number,
                    previous_number_of_samples=number_of_samples,
                )
                self._downsampler.inform_end_of_current_label()
        else:
            batch_number, number_of_samples = self._iterate_dataloader_and_compute_scores(self._train_dataloader)

        selected_ids, weights = self._downsampler.select_points()

        # to store all the selected (sample, weight).
        # TODO(#283) investigate which size performs the best
        file_size = self._num_dataloaders * self._batch_size
        local_dataset = LocalDatasetWriter(
            self.pipeline_id, self.trigger_id, self._num_dataloaders, file_size, self.offline_dataset_path
        )

        # store the selected samples (id and weight)
        local_dataset.inform_samples(sample_ids=selected_ids, sample_weights=weights)

        # samples are automatically stored when the desired file size is reached. Since the last file might be smaller
        # we need to manually trigger the store
        local_dataset.finalize()

        # instead of getting keys from the selector, now are taken from the local storage
        new_key_source = LocalKeySource(
            pipeline_id=self.pipeline_id, trigger_id=self.trigger_id, offline_dataset_path=self.offline_dataset_path
        )
        self._train_dataloader.dataset.change_key_source(new_key_source)

        self.update_queue(AvailableQueues.DOWNSAMPLING, batch_number, number_of_samples, training_active=True)
        # set the model to train
        self._model.model.train()

    def _iterate_dataloader_and_compute_scores(
        self,
        dataloader: torch.utils.data.DataLoader,
        previous_batch_number: int = 0,
        previous_number_of_samples: int = 0,
    ) -> Tuple[int, int]:
        number_of_samples = previous_number_of_samples
        batch_number = previous_batch_number
        for batch_number, batch in enumerate(dataloader):
            self.update_queue(AvailableQueues.DOWNSAMPLING, batch_number, number_of_samples, training_active=False)

            sample_ids, target, data = self.preprocess_batch(batch)
            number_of_samples += len(sample_ids)

            with torch.autocast(self._device_type, enabled=self._amp):
                # compute the scores and accumulate them
                model_output = self._model.model(data)
                self._downsampler.inform_samples(sample_ids, model_output, target)
        return batch_number, number_of_samples

    def end_of_trigger_cleaning(self) -> None:
        self._train_dataloader.dataset.end_of_trigger_cleaning()

    def _get_available_labels_from_selector(self) -> list[int]:
        req = GetAvailableLabelsRequest(pipeline_id=self.pipeline_id)

        response: AvailableLabelsResponse = self.selector_stub.get_available_labels(req)

        return response.available_labels


def train(
    training_info: TrainingInfo,
    device: str,
    log_path: pathlib.Path,
    exception_queue: mp.Queue,
    status_query_queue_training: mp.Queue,
    status_response_queue_training: mp.Queue,
    status_query_queue_downsampling: mp.Queue,
    status_response_queue_downsampling: mp.Queue,
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
        trainer = PytorchTrainer(
            training_info,
            device,
            status_query_queue_training,
            status_response_queue_training,
            status_query_queue_downsampling,
            status_response_queue_downsampling,
            logger,
        )
        trainer.train()
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
        exception_queue.put(exception_msg)
