# pylint: disable=no-name-in-module
from __future__ import annotations

import copy
import glob
import io
import json
import logging
import os
import pathlib
import shutil
import tempfile
import traceback
from typing import Any

import torch
import transformers

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.evaluator.internal.dataset.evaluation_dataset import EvaluationDataset
from modyn.evaluator.internal.utils.tuning_info import TuningInfo
from modyn.models.modular_adapters.modular_adapters import apply_kadapter, apply_lora
from modyn.trainer_server.internal.trainer.gpu_measurement import GPUMeasurement
from modyn.trainer_server.internal.utils.metric_type import MetricType
from modyn.utils import (
    LABEL_TRANSFORMER_FUNC_NAME,
    deserialize_function,
    dynamic_module_import,
    package_available_and_can_be_imported,
    seed_everything,
)


class PytorchTuner:
    # pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches, too-many-statements

    def __init__(self, tuning_info: TuningInfo, device: str, logger: logging.Logger, model: Any) -> None:
        self.logger = logger

        self._info("Initializing Pytorch Tuner")
        self.generative = tuning_info.generative
        self._grad_norm = 0.5  # remember add this to training infotuning_info.grad_norm
        self._lora = False
        self._kadapter = False
        self._light_tuning_steps = tuning_info.steps
        if tuning_info.seed is not None:
            self._seed_trainer_server(tuning_info.seed)
            self._info("Everything seeded")

        # setup model and optimizer
        self._model = model
        self._setup_optimizers(tuning_info)
        self._info("Model and optimizer created.")

        self._scaler = torch.cuda.amp.GradScaler(enabled=tuning_info.amp, **tuning_info.grad_scaler_configuration)
        self._info("Grad scaler created.")
        if self._lora:
            apply_lora(self._model.model)
        if self._kadapter:
            apply_kadapter(self._model.model)

        criterion_func = getattr(torch.nn, tuning_info.torch_criterion)
        self._criterion = criterion_func(**tuning_info.criterion_dict)

        self._batch_size = tuning_info.batch_size
        self._num_dataloaders = tuning_info.num_dataloaders

        self._label_transformer_function = deserialize_function(
            tuning_info.label_transformer, LABEL_TRANSFORMER_FUNC_NAME
        )

        self._device = device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self._amp = tuning_info.amp

        self._measure_gpu_ops = tuning_info.enable_accurate_gpu_measurements

        self.epochs_per_trigger = tuning_info.epochs

        self.pipeline_id = tuning_info.pipeline_id
        self._drop_last_batch = tuning_info.drop_last_batch
        self._dataset_log_path = pathlib.Path(tempfile.mkdtemp(prefix=f"pl{self.pipeline_id}"))
        self._log_file_path = tuning_info.log_file_path
        if self._log_file_path is not None:
            assert isinstance(self._log_file_path, pathlib.Path)
            self._log_file_path.unlink(missing_ok=True)
        else:
            logger.warn("Log file path is None.")

        self._log: dict[str, Any] = {}

        self._num_samples = 0

        self._expected_num_batches = -1
        self._expected_num_epochs = -1

        self._step_lr_every: str | None = None
        self._setup_lr_scheduler(tuning_info)

        self._info("LR scheduler created.")
        self._evaluation_id = tuning_info.evaluation_id

        def _prepare_dataloader(
            tuning_info: TuningInfo,
        ) -> torch.utils.data.DataLoader:
            dataset = EvaluationDataset(
                tuning_info.dataset_id,
                tuning_info.bytes_parser,
                tuning_info.transform_list,
                tuning_info.storage_address,
                tuning_info.evaluation_id,
                tuning_info.tokenizer,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=tuning_info.batch_size,
                num_workers=tuning_info.num_dataloaders,
                timeout=60 if tuning_info.num_dataloaders > 0 else 0,
            )

            return dataloader

        # setup dataloaders
        self._info("Setting up data loaders.")
        self._train_dataloader, self._val_dataloader = _prepare_dataloader(tuning_info)

        # Create callbacks
        # TODO(#140): should be defined by the pipeline and passed with training request
        self._callbacks: dict[MetricType, Any] = {
            # MetricType.LOSS: LossCallback(self._metadata_collector, criterion_func, tuning_info.criterion_dict)
        }

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                       Core training pipeline orchestration                                       #
    # ---------------------------------------------------------------------------------------------------------------- #

    def train(self) -> None:
        """Performs light tuning for a few steps before evaluation."""
        self._info(f"Process {os.getpid()} starts light tuning")

        stopw = Stopwatch()

        stopw.start("TotalLightTuning")
        self._model.model.train()

        for step, batch in enumerate(self._train_dataloader):
            if step >= self._light_tuning_steps:
                break  # Stop after defined steps

            stopw.start("FetchBatch", resume=True)
            sample_ids, target, data = self.preprocess_batch(batch, stopw)

            for _, optimizer in self._optimizers.items():
                optimizer.zero_grad()

            # Forward pass
            with torch.autocast(self._device_type, enabled=self._amp):
                stopw.start("Forward", resume=True)

                if self.generative:
                    output = self._model.model(data)
                    output = output[..., :-1, :]  # Ignore last token prediction
                    target = data[..., 1:, 0]  # Shift target labels

                    output = output.reshape(-1, output.size(-1))
                    target = target.reshape(-1)

                    target[target == 50256] = -100  # Mask padding tokens for GPT-style models

                else:
                    output = self._model.model(data, sample_ids=sample_ids)

                stopw.stop("Forward")

                # Compute loss
                stopw.start("Loss", resume=True)
                loss = self._criterion(output, target)
                stopw.stop("Loss")

            # Backward pass and optimizer step
            stopw.start("Backward", resume=True)
            self._scaler.scale(loss).backward()
            if self._grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self._model.model.parameters(), max_norm=self._grad_norm)
            stopw.stop("Backward")

            stopw.start("OptimizerStep", resume=True)
            for _, optimizer in self._optimizers.items():
                self._scaler.step(optimizer)
            self._scaler.update()
            self._step_lr_if_necessary(True)
            stopw.stop("OptimizerStep")

            # Log loss

        stopw.stop("TotalLightTuning")
        self._info(f"Light tuning complete! Total time: {stopw.measurements.get('TotalLightTuning', 0)} seconds")

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                  Training stages                                                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    def preprocess_batch(
        self, batch: tuple, stopw: Stopwatch | None = None
    ) -> tuple[list, torch.Tensor, torch.Tensor | dict]:
        if stopw is None:
            stopw = Stopwatch()

        stopw.start("PreprocSampleIDs", resume=True)
        sample_ids = batch[0]
        if isinstance(sample_ids, torch.Tensor):
            sample_ids = sample_ids.tolist()
        elif isinstance(sample_ids, tuple):
            sample_ids = list(sample_ids)
        assert isinstance(sample_ids, list), "Cannot parse result from DataLoader"
        stopw.stop("PreprocSampleIDs")
        if self.generative:
            target = None
        else:
            stopw.start("LabelTransform", resume=True)
            if self._label_transformer_function is not None:
                target = self._label_transformer_function(batch[2])
            else:
                target = batch[2]
            stopw.stop("LabelTransform")

            with GPUMeasurement(self._measure_gpu_ops, "MoveLabelToGPU", self._device, stopw, resume=True):
                target = target.to(self._device)

        with GPUMeasurement(self._measure_gpu_ops, "MoveDataToGPU", self._device, stopw, resume=True):
            data: torch.Tensor | dict
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

    def _step_lr_if_necessary(self, is_batch: bool) -> None:
        if self._lr_scheduler is None:
            return
        assert self._step_lr_every is not None  # for mypy

        if is_batch and self._step_lr_every == "batch":
            self._lr_scheduler.step()

        if not is_batch and self._step_lr_every == "epoch":
            self._lr_scheduler.step()

    # ------------------------------------------------------ IO ------------------------------------------------------ #

    def save_state(self, destination: pathlib.Path | io.BytesIO, iteration: int | None = None) -> None:
        dict_to_save = {}
        dict_to_save["model"] = self._model.model.state_dict()
        for optimizer_name, optimizer in self._optimizers.items():
            dict_to_save[f"optimizer-{optimizer_name}"] = optimizer.state_dict()

        if iteration is not None:
            dict_to_save["iteration"] = iteration
        print(destination)
        torch.save(dict_to_save, destination)

    def _setup_optimizers(self, tuning_info: TuningInfo) -> None:
        self._optimizers = {}
        for name, optimizer_config in tuning_info.torch_optimizers_configuration.items():
            if optimizer_config["source"] == "PyTorch":
                optimizer_func = getattr(torch.optim, optimizer_config["algorithm"])
            elif optimizer_config["source"] == "APEX":
                if package_available_and_can_be_imported("apex"):
                    import apex  # pylint: disable=import-outside-toplevel, import-error

                    optimizer_func = getattr(apex.optimizers, optimizer_config["algorithm"])
                else:
                    raise ValueError("Apex Optimizer defined, but apex is not available in the system")
            elif optimizer_config["source"] == "HuggingFace":
                optimizer_func = getattr(transformers, optimizer_config["algorithm"])
            else:
                raise ValueError(
                    f"Unsupported optimizer from {optimizer_config['source']}. PyTorch and APEX are supported"
                )
            optimizer_config_list = []
            for param_group in optimizer_config["param_groups"]:
                module = param_group["module"]

                if optimizer_config["algorithm"] == "Adafactor":  # Check if optimizer is Adafactor
                    # Debug: Print the type of self._model
                    no_decay = ["bias", "LayerNorm.weight"]

                    # Create separate parameter group dictionaries
                    param_group_no_decay = copy.deepcopy(param_group["config"])
                    param_group_decay = copy.deepcopy(param_group["config"])

                    param_group_decay["params"] = [
                        p
                        for n, p in eval(f"self._model.{module}.named_parameters()")  # pylint: disable=eval-used
                        if not any(m in n for m in no_decay)
                    ]
                    param_group_decay["weight_decay"] = 0.01
                    optimizer_config_list.append(param_group_decay)

                    param_group_no_decay["params"] = [
                        p
                        for n, p in eval(f"self._model.{module}.named_parameters()")  # pylint: disable=eval-used
                        if any(m in n for m in no_decay)
                    ]

                    param_group_no_decay["weight_decay"] = 0.0
                    optimizer_config_list.append(param_group_no_decay)

                else:
                    param_group["config"]["params"] = eval(  # pylint: disable=eval-used
                        f"self._model.{module}.parameters()"
                    )

                    optimizer_config_list.append(param_group["config"])
            self._optimizers[name] = optimizer_func(optimizer_config_list)

    def _update_lr_config_dict(self, lr_scheduler_config: dict[str, Any]) -> dict[str, Any]:
        for key, value in lr_scheduler_config.items():
            if isinstance(value, dict):
                self._update_lr_config_dict(value)
            elif value == "MODYN_NUM_BATCHES":
                lr_scheduler_config[key] = self._expected_num_batches
            elif value == "MODYN_NUM_EPOCHS":
                lr_scheduler_config[key] = self._expected_num_epochs

        return lr_scheduler_config

    def _setup_lr_scheduler(self, tuning_info: TuningInfo) -> None:
        self._lr_scheduler = None
        if tuning_info.lr_scheduler:
            self._step_lr_every = tuning_info.lr_scheduler["step_every"]

            config_dict = self._update_lr_config_dict(tuning_info.lr_scheduler["config"])

            if tuning_info.lr_scheduler["source"] == "Custom":
                lr_scheduler_module = dynamic_module_import("modyn.trainer_server.custom_lr_schedulers")
                custom_lr_scheduler = getattr(lr_scheduler_module, tuning_info.lr_scheduler["name"])
                optimizers = [self._optimizers[opt] for opt in tuning_info.lr_scheduler["optimizers"]]
                self._lr_scheduler = custom_lr_scheduler(optimizers, config_dict)
            elif tuning_info.lr_scheduler["source"] == "PyTorch":
                torch_lr_scheduler = getattr(torch.optim.lr_scheduler, tuning_info.lr_scheduler["name"])
                if len(tuning_info.lr_scheduler["optimizers"]) > 1:
                    self._warning("Provided a LR scheduler from PyTorch, but multiple optimizers")
                self._lr_scheduler = torch_lr_scheduler(
                    self._optimizers[tuning_info.lr_scheduler["optimizers"][0]],
                    **config_dict,
                )
            else:
                raise ValueError(
                    f"Unsupported LR scheduler of source {tuning_info.lr_scheduler['source']}."
                    "PyTorch and Custom are supported"
                )

    def _seed_trainer_server(self, seed: int) -> None:
        if not (0 <= seed <= 100 and isinstance(seed, int)):
            raise ValueError("The seed must be an integer in the range [0,100]")
        # seed the trainer server
        seed_everything(seed)

    # ---------------------------------------------------- Logging --------------------------------------------------- #

    def _info(self, msg: str) -> None:
        self.logger.info(f"[Training {self._evaluation_id}][PL {self.pipeline_id}] {msg}")

    def _warning(self, msg: str) -> None:
        self.logger.warning(f"[Training {self._evaluation_id}][PL {self.pipeline_id}] {msg}")

    def _error(self, msg: str) -> None:
        self.logger.error(f"[Training {self._evaluation_id}][PL {self.pipeline_id}] {msg}")

    def _load_dataset_log(self) -> None:
        worker_log = {}
        for filename in glob.glob(str(self._dataset_log_path / "*.log")):
            filepath = pathlib.Path(filename)
            key = filepath.stem

            with open(self._dataset_log_path / filename, encoding="utf-8") as logfile:
                worker_log[key] = json.load(logfile)

        self._log["dataset_worker_log"] = worker_log

        try:
            if self._dataset_log_path.exists():
                shutil.rmtree(self._dataset_log_path)
        except OSError as exp:
            self._error("Error while deleting OnlineDataset logging directory.")
            self._error(str(exp))

    # -------------------------------------------------- Assertions -------------------------------------------------- #

    @staticmethod
    def _assert_data_size(
        expected_size: int, data: torch.Tensor | dict[Any, torch.Tensor], sample_ids: list, target: torch.Tensor
    ) -> None:
        assert (
            all(tensor.shape[0] == expected_size for tensor in data.values())
            if isinstance(data, dict)
            else data.shape[0] == expected_size
        ), (
            f"expected size: {expected_size}, actual size: "
            + f"{data.shape[0] if isinstance(data, torch.Tensor) else 'n/a'}"
        )
        assert len(sample_ids) == expected_size, f"expected size: {expected_size}, actual size: {len(sample_ids)}"
        assert target.shape[0] == expected_size, f"expected size: {expected_size}, actual size: {target.shape[0]}"

    def _assert_training_size(self, epoch: int, trained_batches: int) -> None:
        if self._lr_scheduler is not None:
            assert self._expected_num_epochs == epoch + 1, (
                f"Something went wrong! We expected {self._expected_num_epochs}, but trained for {epoch + 1} epochs!"
                + "\nWe fail since we trained using a LR scheduler that might depend on this."
            )
            assert self._expected_num_batches == trained_batches, (
                f"Something went wrong! We expected to train on {self._expected_num_batches},"
                + f" but trained for {trained_batches} batches!"
                + "\nWe fail since we trained using a LR scheduler that might depend on this."
            )
        else:
            if self._expected_num_epochs != epoch + 1 or self._expected_num_batches != trained_batches:
                self._error(
                    "Inconsistent expected batches. Not failing since no lr scheduler was used.\n"
                    + f" We expected {self._expected_num_epochs}, but trained for {epoch + 1} epochs!\n"
                    + f"We expected to train on {self._expected_num_batches},"
                    + f" but trained for {trained_batches} batches!"
                )

    # ---------------------------------------------------- Cleanup --------------------------------------------------- #


def tune(
    tuning_info: TuningInfo,
    device: str,
    log_path: pathlib.Path,
    model: Any,
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
        tuner = PytorchTuner(tuning_info, device, logger, model)
        tuner.train()
    except Exception:  # pylint: disable=broad-except
        exception_msg = traceback.format_exc()
        logger.error(exception_msg)
