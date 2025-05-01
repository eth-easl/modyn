# pylint: disable=no-name-in-module
from __future__ import annotations

import contextlib
import copy
import glob
import io
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
import pathlib
import queue
import shutil
import tempfile
import traceback
from collections.abc import Iterable
from typing import Any, Literal

import grpc
import numpy as np
import torch
import transformers

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.models.coreset_methods_support import CoresetSupportingModule
from modyn.models.dlrm.dlrm import DLRM
from modyn.models.modular_adapters.modular_adapters import apply_adapters
from modyn.models.t5.t5 import T5
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    AvailableLabelsResponse,
    GetAvailableLabelsRequest,
    GetNumberOfSamplesRequest,
    GetSelectionStrategyRequest,
    NumberOfSamplesResponse,
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
from modyn.trainer_server.internal.trainer.batch_accumulator import BatchAccumulator
from modyn.trainer_server.internal.trainer.gpu_measurement import GPUMeasurement
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_per_label_remote_downsample_strategy import (
    AbstractPerLabelRemoteDownsamplingStrategy,
)
from modyn.trainer_server.internal.trainer.remote_downsamplers.abstract_remote_downsampling_strategy import (
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
    instantiate_class,
    package_available_and_can_be_imported,
    seed_everything,
)

AvailableQueues = Literal["TRAINING", "DOWNSAMPLING"]


class PytorchTrainer:
    # pylint: disable=too-many-instance-attributes, too-many-locals, too-many-branches, too-many-statements

    def __init__(
        self,
        modyn_config: dict,
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
        self._info("Initializing Pytorch Trainer")

        self.training_type = training_info.training_type
        self._grad_norm = training_info.grad_norm
        self.gradient_accumulation_steps = training_info.gradient_accumulation_steps

        self.selector_stub = self.connect_to_selector(training_info.selector_address)
        if training_info.seed is not None:
            self._seed_trainer_server(training_info.seed)
            self._info("Everything seeded")

        self._info("Initializing model...")
        self._model = training_info.model_handler(training_info.model_configuration_dict, device, training_info.amp)
        self._info("Model and optimizer created.")

        self._scaler = torch.cuda.amp.GradScaler(enabled=training_info.amp, **training_info.grad_scaler_configuration)
        self._info("Grad scaler created.")

        self._model.model = apply_adapters(
            self._model.model, training_info.model_wrappers, training_info.model_wrapper_args
        )

        self.expert_mixture = False
        self._setup_optimizers(training_info)

        if training_info.use_pretrained_model:
            self._info("Loading model state from pretrained model.")
            self.load_state_if_given(training_info.pretrained_model_path, training_info.load_optimizer_state)

        self._model.model.to(device)
        criterion_func = getattr(torch.nn, training_info.torch_criterion)
        self._criterion = criterion_func(**training_info.criterion_dict)

        self._batch_size = training_info.batch_size
        self._num_dataloaders = training_info.num_dataloaders
        self._device = device
        self._device_type = "cuda" if "cuda" in self._device else "cpu"
        self._amp = training_info.amp
        self._measure_gpu_ops = training_info.enable_accurate_gpu_measurements

        self._label_transformer_function = deserialize_function(
            training_info.label_transformer, LABEL_TRANSFORMER_FUNC_NAME
        )

        self._checkpoint_path = training_info.checkpoint_path
        self._checkpoint_interval = training_info.checkpoint_interval
        self._record_loss_every = training_info.record_loss_every
        self._final_checkpoint_path = training_info.final_checkpoint_path
        self.epochs_per_trigger = training_info.epochs_per_trigger
        self.num_samples_to_pass = training_info.num_samples_to_pass
        self._log_file_path = training_info.log_file_path
        self._drop_last_batch = training_info.drop_last_batch

        self._dataset_log_path = pathlib.Path(tempfile.mkdtemp(prefix=f"pl{self.pipeline_id}"))
        if not self._checkpoint_path.is_dir():
            self._checkpoint_path.mkdir()
        self._final_checkpoint_path.mkdir()

        if self._log_file_path is not None:
            assert isinstance(self._log_file_path, pathlib.Path)
            self._log_file_path.unlink(missing_ok=True)
        else:
            logger.warn("Log file path is None.")

        self._log: dict[str, Any] = {}

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
            self._setup_downsampling(criterion_func, downsampler_config, modyn_config, strategy_name, training_info)
        else:
            self._downsampling_mode = DownsamplingMode.DISABLED

        self._expected_num_batches = -1
        self._expected_num_epochs = -1
        self._calc_expected_sizes(downsampling_enabled)

        self._step_lr_every: str | None = None
        self._setup_lr_scheduler(training_info)
        self._info("LR scheduler created.")

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
            training_info.num_prefetched_partitions,
            training_info.parallel_prefetch_requests,
            training_info.shuffle,
            training_info.tokenizer,
            self._dataset_log_path,
            drop_last=self._drop_last_batch,
            include_labels=training_info.training_type != "generative",
            transform_target=training_info.transform_target,
            bytes_parser_target=training_info.bytes_parser_target,
            seq_length=training_info.tokenizer_seq_length,
        )

        self._callbacks: dict[MetricType, Any] = {
            # MetricType.LOSS: LossCallback(self._metadata_collector, criterion_func, training_info.criterion_dict)
        }

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                       Core training pipeline orchestration                                       #
    # ---------------------------------------------------------------------------------------------------------------- #

    def train(self) -> None:  # pylint: disable=too-many-locals, too-many-branches
        self._info(f"Process {os.getpid()} starts training")
        total_stopw = Stopwatch()
        stopw = Stopwatch()
        total_stopw.start("TotalTrain")

        self._model.model.train()

        stopw.start("OnBeginCallbacks")
        for _, callback in self._callbacks.items():
            callback.on_train_begin(self._model.model, self._optimizers)
        self._log["on_begin_callbacks_time"] = stopw.stop()

        self._info("Handled OnBegin Callbacks.")
        self._log["epochs"] = []
        training_loss: list[float] = []
        if self.num_samples_to_pass == 0:
            epoch_num_generator: Iterable[int] = range(self.epochs_per_trigger)
        else:
            # an infinity epoch generator
            epoch_num_generator = itertools.count(start=0)
            self._info(f"Training will stop when the number of samples to pass reaches {self.num_samples_to_pass}.")

        if self._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
            # assertion since model validation by pydantic should catch this.
            assert self._downsampler.supports_bts, "The downsampler does not support batch then sample"
            # We cannot pass the target size from the trainer server since that depends on StB vs BtS.
            post_downsampling_size = max(
                (self._downsampler.downsampling_ratio * self._batch_size) // self._downsampling_ratio_max, 1
            )
     
            assert post_downsampling_size <= self._batch_size
            if self._batch_size % post_downsampling_size != 0:
                raise ValueError(
                    f"The target batch size of {self._batch_size} is not a multiple of the batch size "
                    + f"after downsampling a batch in BtS mode ({post_downsampling_size}). We cannot accumulate "
                    + "batches. Please choose the downsampling ratio and batch size such that this is possible."
                )
            batch_accumulator = BatchAccumulator(self._batch_size // post_downsampling_size, self._device)

        trained_batches = 0
        passed_batches = 0
        for epoch in epoch_num_generator:
            stopw = Stopwatch()  # Reset timings per epoch
            self._log["epochs"].append({})
            batch_timings = []

            if self._sample_then_batch_this_epoch(epoch):
                self.update_queue(
                    "TRAINING", trained_batches, trained_batches * self._batch_size, training_active=False
                )
                with GPUMeasurement(self._measure_gpu_ops, "DownsampleSTB", self._device, stopw):
                    self.downsample_trigger_training_set()

            stopw.start("IndivFetchBatch", overwrite=True)
            stopw.start("FetchBatch", resume=True)

            accumulation_counter = 0  # NEW: Initialize accumulation counter.
            for batch in self._train_dataloader:
                stopw.stop("FetchBatch")
                batch_timings.append(stopw.stop("IndivFetchBatch"))
                retrieve_weights_from_dataloader, weighted_optimization = self.weights_handling(len(batch))

                stopw.start("OnBatchBeginCallbacks", resume=True)
                for _, callback in self._callbacks.items():
                    callback.on_batch_begin(self._model.model, self._optimizers, batch, passed_batches)
                stopw.stop()

                self.update_queue("TRAINING", trained_batches, trained_batches * self._batch_size, training_active=True)
                passed_batches += 1
                with GPUMeasurement(self._measure_gpu_ops, "PreprocessBatch", self._device, stopw, resume=True):
                    sample_ids, target, data = self.preprocess_batch(batch, stopw)

                if retrieve_weights_from_dataloader:
                    # model output is a torch.FloatTensor but weights is a torch.DoubleTensor.
                    # We need to cast to do the dot product
                    weights = batch[3].float().to(self._device)
                if accumulation_counter == 0:  # zero grad is moved here
                    for _, optimizer in self._optimizers.items():
                        optimizer.zero_grad()

                with torch.autocast(self._device_type, enabled=self._amp):
                    if self._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE:
                        with GPUMeasurement(self._measure_gpu_ops, "DownsampleBTS", self._device, stopw, resume=True):
                            data, sample_ids, target, weights = self.downsample_batch(data, sample_ids, target)

                        self._assert_data_size(post_downsampling_size, data, sample_ids, target)
                        if not batch_accumulator.inform_batch(data, sample_ids, target, weights):
                            stopw.start("FetchBatch", resume=True)
                            stopw.start("IndivFetchBatch", overwrite=True)
                            self._num_samples += self._batch_size
                            continue

                        data, sample_ids, target, weights = batch_accumulator.get_accumulated_batch()

                        self._assert_data_size(self._batch_size, data, sample_ids, target)

                    with GPUMeasurement(self._measure_gpu_ops, "Forward", self._device, stopw, resume=True):
                        # Measure memory usage before forward pass
                        # initial_memory = torch.cuda.memory_allocated()
                        # print(f"Before forward pass: {initial_memory / 1e9:.2f} GB")
                        # torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking
                        if self.training_type == "generative":
                            output = self._model.model(data, labels=target)
                            
                        elif self.training_type == "pretraining":
                            output = self._model.model(data)

                        else:
                            # Non-generative task: Pass data, and optionally sample_ids if required
                            output = self._model.model(data, sample_ids=sample_ids)

                        # Measure memory usage after forward pass
                        # final_memory = torch.cuda.memory_allocated()
                        # peak_memory = torch.cuda.max_memory_allocated()
                        # print(f"After forward pass: {final_memory / 1e9:.2f} GB")
                        # print(f"Peak memory during forward pass: {peak_memory / 1e9:.2f} GB")

                    with GPUMeasurement(self._measure_gpu_ops, "Loss", self._device, stopw, resume=True):
                        # Measure memory usage before loss computation
                        # initial_memory = torch.cuda.memory_allocated()
                        # print(f"Before loss computation: {initial_memory / 1e9:.2f} GB")
                        torch.cuda.reset_peak_memory_stats()  # Reset peak memory tracking

                        if self.training_type != "labeled":
                            # Shift logits and labels for next-token prediction
                            # output = output.logits[..., :, :]  # Output for all tokens except the last one
                            # Target for all tokens except the first one
                            if output.size(1) > target.size(1):
                                diff = output.size(1) - target.size(1)
                                pad_tensor = torch.full(
                                    (target.size(0), diff), -100, dtype=target.dtype, device=target.device
                                )
                                target = torch.cat([target,pad_tensor], dim=1)

                            # Use reshape instead of view to handle non-contiguous tensors safely
                            output = output.reshape(-1, output.size(-1))
                            target = target.reshape(-1)

                            # Calculate loss
                            if weighted_optimization:
                                # Compute per-timestep loss (currently 1D, shape: [L])
                                loss_per_timestep = self._criterion_nored(output, target)
                                batch_size = weights.size(0)
                                seq_length = loss_per_timestep.numel() // batch_size
                                loss_per_timestep = loss_per_timestep.reshape(batch_size, seq_length)
                                normalized_weights = weights / weights.sum()
                                expanded_weights = normalized_weights.unsqueeze(1).expand(batch_size, seq_length)
                                loss = torch.dot(loss_per_timestep.reshape(-1), expanded_weights.reshape(-1))
                            
                            else:
                                loss = self._criterion(output, target)
                        else:
                            if weighted_optimization:
                                # Weighted gradient descent
                                assert weights is not None
                                loss = torch.dot(self._criterion_nored(output, target), weights / weights.sum())
                            else:
                                loss = self._criterion(output, target)
                        loss = loss / self.gradient_accumulation_steps  # Scale loss for gradient accumulation

                        # Measure memory usage after loss computation
                        # final_memory = torch.cuda.memory_allocated()
                        # peak_memory = torch.cuda.max_memory_allocated()
                        # print(f"After loss computation: {final_memory / 1e9:.2f} GB")
                        # print(f"Peak memory during loss computation: {peak_memory / 1e9:.2f} GB")

                stopw.start("OnBatchBeforeUpdate", resume=True)
                for _, callback in self._callbacks.items():
                    callback.on_batch_before_update(
                        self._model.model, self._optimizers, trained_batches, sample_ids, data, target, output, loss
                    )
                stopw.stop()

                with GPUMeasurement(self._measure_gpu_ops, "Backward", self._device, stopw, resume=True):
                    self._scaler.scale(loss).backward()
                    if self._grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self._model.model.parameters(), max_norm=self._grad_norm)
                accumulation_counter += 1
                trained_batches += 1
                if accumulation_counter == self.gradient_accumulation_steps:
                    with GPUMeasurement(self._measure_gpu_ops, "OptimizerStep", self._device, stopw, resume=True):
                        for _, optimizer in self._optimizers.items():
                            self._scaler.step(optimizer)
                        self._scaler.update()

                    accumulation_counter = 0  # NEW: Reset accumulation counter.
                    self._step_lr_if_necessary(True)  # NEW: Step LR scheduler after optimizer update.
                    if self._checkpoint_interval > 0 and trained_batches % self._checkpoint_interval == 0:
                        stopw.start("Checkpoint", resume=True)
                        checkpoint_file_name = self._checkpoint_path / f"model_{trained_batches}.modyn"
                        self.save_state(checkpoint_file_name, trained_batches)
                        stopw.stop("Checkpoint")
                    if self._record_loss_every > 0 and trained_batches % self._record_loss_every == 0:
                        training_loss.append(loss.item())
                        print(f"Training loss: {loss.item()}")  # I think this is generally useful for debugging.
                        # Log loss and batch number
                        log_file = self._checkpoint_path / "training_log.txt"
                        with (
                            open(log_file, "a") as f  # pylint: disable=unspecified-encoding
                        ):  # 'a' mode appends if the file exists, else creates it
                            f.write(f"{trained_batches},{loss.item()}\n")
                        # Example: Logging training losses in a loop

                    self._num_samples += len(sample_ids)

                    stopw.start("OnBatchEnd", resume=True)
                    for _, callback in self._callbacks.items():
                        callback.on_batch_end(
                            self._model.model, self._optimizers, trained_batches, sample_ids, data, target, output, loss
                        )
                    stopw.stop()
                    if 0 < self.num_samples_to_pass <= self._num_samples:
                        self._info("Stopping training as we have reached the sample threshold.")
                        break

                stopw.start("FetchBatch", resume=True)
                stopw.start("IndivFetchBatch", overwrite=True)
            self._step_lr_if_necessary(False)

            if len(batch_timings) <= 100000:
                self._log["epochs"][epoch]["BatchTimings"] = batch_timings

            # mypy cannot handle np.min and np.max
            if len(batch_timings) > 0:
                batch_timings = np.array(batch_timings)
                self._log["epochs"][epoch]["MinFetchBatch"] = np.min(batch_timings).item()  # type: ignore
                self._log["epochs"][epoch]["MaxFetchBatch"] = np.max(batch_timings).item()  # type: ignore
                self._log["epochs"][epoch]["AvgFetchBatch"] = np.mean(batch_timings).item()
                self._log["epochs"][epoch]["MedianFetchBatch"] = np.median(batch_timings).item()
                self._log["epochs"][epoch]["StdFetchBatch"] = np.std(batch_timings).item()
                del batch_timings
            else:
                self._error("Got zero batch timings, cannot get minimum.")

            self._log["epochs"][epoch]["TotalFetchBatch"] = stopw.measurements.get("FetchBatch", 0)
            self._log["epochs"][epoch]["OnBatchBeginCallbacks"] = stopw.measurements.get("OnBatchBeginCallbacks", 0)
            self._log["epochs"][epoch]["PreprocessBatch"] = stopw.measurements.get("PreprocessBatch", 0)
            self._log["epochs"][epoch]["PreprocSampleIDs"] = stopw.measurements.get("PreprocSampleIDs", 0)
            self._log["epochs"][epoch]["LabelTransform"] = stopw.measurements.get("LabelTransform", 0)
            self._log["epochs"][epoch]["MoveLabelToGPU"] = stopw.measurements.get("MoveLabelToGPU", 0)
            self._log["epochs"][epoch]["MoveDataToGPU"] = stopw.measurements.get("MoveDataToGPU", 0)
            self._log["epochs"][epoch]["DownsampleBTS"] = stopw.measurements.get("DownsampleBTS", 0)
            self._log["epochs"][epoch]["DownsampleSTB"] = stopw.measurements.get("DownsampleSTB", 0)
            self._log["epochs"][epoch]["Forward"] = stopw.measurements.get("Forward", 0)
            self._log["epochs"][epoch]["Loss"] = stopw.measurements.get("Loss", 0)
            self._log["epochs"][epoch]["OnBatchBeforeUpdate"] = stopw.measurements.get("OnBatchBeforeUpdate", 0)
            self._log["epochs"][epoch]["Backward"] = stopw.measurements.get("Backward", 0)
            self._log["epochs"][epoch]["OptimizerStep"] = stopw.measurements.get("OptimizerStep", 0)
            self._log["epochs"][epoch]["Checkpoint"] = stopw.measurements.get("Checkpoint", 0)
            self._log["epochs"][epoch]["OnBatchEnd"] = stopw.measurements.get("OnBatchEnd", 0)

            self._persist_pipeline_log()
            if 0 < self.num_samples_to_pass <= self._num_samples:
                self._info("reached the threshold of samples to pass; break out of epoch loop to stop training")
                break

        total_stopw.stop("TotalTrain")

        self._info(f"Finished training: {self._num_samples} samples, {passed_batches} batches.")
        self._log["num_samples"] = self._num_samples
        self._log["num_samples_trained"] = trained_batches * self._batch_size
        self._log["num_batches"] = passed_batches
        self._log["num_batches_trained"] = trained_batches
        self._log["total_train"] = total_stopw.measurements.get("TotalTrain", 0)
        self._log["training_loss"] = training_loss

        self._assert_training_size(epoch, trained_batches)
        self._load_dataset_log()
        self._persist_pipeline_log()

        for _, callback in self._callbacks.items():
            callback.on_train_end(self._model.model, self._optimizers, self._num_samples, passed_batches)

        for metric in self._callbacks:
            self._metadata_collector.send_metadata(metric)
        self._metadata_collector.cleanup()

        # save final model
        final_checkpoint_file_name = self._final_checkpoint_path / "model_final.modyn"
        self.save_state(final_checkpoint_file_name)

        # clean temporary directories in dataloader
        self.end_of_trigger_cleaning()

        self._info("Training complete!")
        self._persist_pipeline_log()

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                  Training stages                                                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    # --------------------------------------------- Core training stages --------------------------------------------- #

    def downsample_trigger_training_set(self) -> None:
        """Function to score every datapoint in the current PRESAMPLED DATASET
        and sample a fraction of it Used for downsampling strategies in
        SAMPLE_THEN_BATCH mode."""
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

        self.start_embedding_recording_if_needed()

        if self._downsampler.requires_data_label_by_label:
            assert isinstance(self._downsampler, AbstractPerLabelRemoteDownsamplingStrategy)
            available_labels = self.get_available_labels_from_selector()

            number_of_samples = 0
            batch_number = -1
            first_label = True
            for label in available_labels:
                if first_label:
                    per_class_dataloader = prepare_per_class_dataloader_from_online_dataset(
                        self._train_dataloader.dataset,
                        self._batch_size,
                        self._num_dataloaders,
                        label,
                        drop_last=self._drop_last_batch,
                    )
                    first_label = False
                else:
                    assert per_class_dataloader is not None
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

        self.end_embedding_recorder_if_needed()

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

        self.update_queue("DOWNSAMPLING", batch_number + 1, number_of_samples, training_active=True)
        # set the model to train
        self._model.model.train()

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

        with GPUMeasurement(self._measure_gpu_ops, "MoveLabelToGPU", self._device, stopw, resume=True):
            if self.training_type == "generative":
                target: torch.Tensor | dict
                if isinstance(batch[2], torch.Tensor):
                    target = batch[2].to(self._device)
                elif isinstance(batch[2], dict):
                    target: dict[str, torch.Tensor] = {}  # type: ignore[no-redef]
                    for name, tensor in batch[2].items():
                        target[name] = tensor.to(self._device)
                else:
                    raise ValueError(
                        "The format of the data provided is not supported in modyn. "
                        "Please use either torch tensors or dict[str, torch.Tensor]"
                    )
            else:
                stopw.start("LabelTransform", resume=True)

                if self._label_transformer_function is not None:
                    target = self._label_transformer_function(batch[2])
                else:
                    target = batch[2]

                stopw.stop("LabelTransform")
                target = target.to(self._device)
            if self.training_type != "labeled":
                target = target[:, :, 0]
                target[target == self._model.model.tokenizer.pad_token_id] = -100

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

    def is_inference_tensor(self, t: torch.Tensor) -> bool:
        return getattr(t, "_has_inference_meta", False)

    def downsample_batch(
        self, data: dict[str, torch.Tensor] | torch.Tensor, sample_ids: list, target: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor] | torch.Tensor, list, torch.Tensor, torch.Tensor]:
        """Function to score every datapoint in the current BATCH and sample a
        fraction of it Used for downsampling strategies in BATCH_THEN_SAMPLE
        mode.

        Receives the samples, the sample ids and the targets. Returns
        the selected subset of these tensors and the weights for each
        sample.
        """
        assert not self.is_inference_tensor(data), "Found an inference tensor!"
        assert self._downsampler is not None
        assert self._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE
        self._model.model.eval()
        self._downsampler.init_downsampler()
        self.start_embedding_recording_if_needed()

        # DLRM does not support inference_mode(), as it will complain during training that
        # "that inference tensors cannot be saved for backward".
        # It could be that some DLRM parameters are lazily created during the
        # first forward pass and hence they are created as inference tensors if inference mode is used here.
        # If this becomes a problem for more models, we might want to make it a field on the model class instead.
        # Some Huggingface models also show strange behavior inference_mode() because of the way they are initialized
        no_grad_mgr = (
            torch.no_grad() if isinstance(self._model, DLRM) or isinstance(self._model, T5) else torch.inference_mode()
        )
        context_manager = contextlib.nullcontext() if self._downsampler.requires_grad else no_grad_mgr

        with context_manager:
            if isinstance(self._model, T5):
                big_batch_output = (
                    self._model.model(data, labels=target) if self._downsampler.forward_required else torch.Tensor()
                )
                big_batch_output = big_batch_output if self._downsampler.forward_required else torch.Tensor()

            else:
                big_batch_output = self._model.model(data) if self._downsampler.forward_required else torch.Tensor()
            embeddings = self.get_embeddings_if_recorded()
            self._downsampler.inform_samples(sample_ids, data, big_batch_output, target, embeddings)

        self.end_embedding_recorder_if_needed()
        torch.set_printoptions(threshold=10_000)

        # TODO(#218) Persist information on the sample IDs/weights when downsampling is performed
        selected_indexes, weights = self._downsampler.select_points()
        selected_data, selected_target = get_tensors_subset(selected_indexes, data, target, sample_ids)
        sample_ids, data, target = selected_indexes, selected_data, selected_target
        # TODO(#219) Investigate if we can avoid 2 forward passes
        self._model.model.train()

        return data, sample_ids, target, weights.to(self._device)

    def start_embedding_recording_if_needed(self) -> None:
        if self._downsampler.requires_coreset_supporting_module:
            # enable the embedding recorder to keep track of last layer embedding. The embeddings are stored
            # in self._model.model.embedding_recorder.embedding
            assert isinstance(self._model.model, CoresetSupportingModule)
            self._model.model.embedding_recorder.start_recording()

    def get_embeddings_if_recorded(self) -> torch.Tensor | None:
        # supply the embeddings if required by the downsampler
        if self._downsampler.requires_coreset_supporting_module:
            embeddings = self._model.model.embedding_recorder.embedding
        else:
            embeddings = None
        return embeddings

    def end_embedding_recorder_if_needed(self) -> None:
        if self._downsampler.requires_coreset_supporting_module:
            # turn off the embedding recording (not needed for regular training)
            assert isinstance(self._model.model, CoresetSupportingModule)
            self._model.model.embedding_recorder.end_recording()

    def weights_handling(self, batch_len: int) -> tuple[bool, bool]:
        # whether the dataloader returned the weights.
        retrieve_weights_from_dataloader = batch_len == 4  # key, sample, label, weight

        # we want to use weighted optimization if we get weights from the dataloader or if we compute them in the
        # training loop (BATCH_THEN_SAMPLE downsampling mode)
        weighted_optimization = (
            retrieve_weights_from_dataloader or self._downsampling_mode == DownsamplingMode.BATCH_THEN_SAMPLE
        )

        return retrieve_weights_from_dataloader, weighted_optimization

    def _step_lr_if_necessary(self, is_batch: bool) -> None:
        if self._lr_scheduler is None:
            return
        assert self._step_lr_every is not None  # for mypy

        if is_batch and self._step_lr_every == "batch":
            self._lr_scheduler.step()

        if not is_batch and self._step_lr_every == "epoch":
            self._lr_scheduler.step()

    # ------------------------------------------------------ IO ------------------------------------------------------ #

    def connect_to_selector(self, selector_address: str) -> SelectorStub:
        selector_channel = grpc.insecure_channel(selector_address)
        assert selector_channel is not None
        if not grpc_connection_established(selector_channel):
            raise ConnectionError(f"Could not establish gRPC connection to selector at address {selector_address}.")
        return SelectorStub(selector_channel)

    def get_available_labels_from_selector(self) -> list[int]:
        req = GetAvailableLabelsRequest(pipeline_id=self.pipeline_id)

        response: AvailableLabelsResponse = self.selector_stub.get_available_labels(req)

        return response.available_labels

    def send_model_state_to_server(self) -> None:
        buffer = io.BytesIO()
        self.save_state(buffer)
        buffer.seek(0)
        bytes_state = buffer.read()
        self._status_response_queue_training.put(bytes_state)

    def get_selection_strategy(self) -> tuple[bool, str, dict]:
        req = GetSelectionStrategyRequest(pipeline_id=self.pipeline_id)

        response: SelectionStrategyResponse = self.selector_stub.get_selection_strategy(req)
        downsampler_config = json.loads(response.downsampler_config.value)

        return response.downsampling_enabled, response.strategy_name, downsampler_config

    def update_queue(
        self, queue_name: AvailableQueues, batch_number: int, number_of_samples: int, training_active: bool
    ) -> None:
        if queue_name == "TRAINING":
            queue_in = self._status_query_queue_training
            queue_out = self._status_response_queue_training
        elif queue_name == "DOWNSAMPLING":
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

    def get_num_samples_in_trigger(self) -> int:
        assert self.selector_stub is not None

        req = GetNumberOfSamplesRequest(pipeline_id=self.pipeline_id, trigger_id=self.trigger_id)
        res: NumberOfSamplesResponse = self.selector_stub.get_number_of_samples(req)

        return res.num_samples

    # ----------------------------------------------- State management ----------------------------------------------- #

    def save_state(self, destination: pathlib.Path | io.BytesIO, iteration: int | None = None) -> None:
        dict_to_save = {}
        dict_to_save["model"] = self._model.model.state_dict()
        for optimizer_name, optimizer in self._optimizers.items():
            dict_to_save[f"optimizer-{optimizer_name}"] = optimizer.state_dict()

        if iteration is not None:
            dict_to_save["iteration"] = iteration

        torch.save(dict_to_save, destination)

    def load_state_if_given(self, path: pathlib.Path | None, load_optimizer_state: bool = False) -> None:
        if path is None:
            return
        assert path.exists(), "Cannot load state from non-existing file"
        self._info(f"Loading model state from {path}")
        # We load the weights on the CPU, and `load_state_dict` moves them to GPU
        with open(path, "rb") as state_file:
            checkpoint = torch.load(io.BytesIO(state_file.read()), map_location=torch.device("cpu"))

        assert "model" in checkpoint
        self._model.model.load_state_dict(checkpoint["model"])
        if load_optimizer_state:
            for optimizer_name, optimizer in self._optimizers.items():
                if f"optimizer-{optimizer_name}" in checkpoint:
                    optimizer.load_state_dict(checkpoint[f"optimizer-{optimizer_name}"])

        os.remove(path)

    # ---------------------------------------------------------------------------------------------------------------- #
    #                                                     Internal                                                     #
    # ---------------------------------------------------------------------------------------------------------------- #

    # ----------------------------------------------------- Setup ---------------------------------------------------- #

    def _persist_pipeline_log(self) -> None:
        if "PYTEST_CURRENT_TEST" in os.environ:
            json.dumps(self._log)  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        if self._log_file_path is not None:
            with open(self._log_file_path, "w", encoding="utf-8") as logfile:
                json.dump(self._log, logfile)
        else:
            self.logger.error("Log file path is None, cannot persist.")

    def _setup_downsampling(
        self,
        criterion_func: torch.nn.modules.loss,
        downsampler_config: dict,
        modyn_config: dict,
        strategy_name: str,
        training_info: TrainingInfo,
    ) -> None:
        self._criterion_nored = criterion_func(**training_info.criterion_dict, reduction="none")
        self._downsampler = self._instantiate_downsampler(
            strategy_name, downsampler_config, modyn_config, self._criterion_nored
        )
        self._downsampling_ratio_max = downsampler_config["ratio_max"]
        assert "sample_then_batch" in downsampler_config
        self._log["received_downsampler_config"] = downsampler_config
        if downsampler_config["sample_then_batch"]:
            self._downsampling_mode = DownsamplingMode.SAMPLE_THEN_BATCH
            assert "downsampling_period" in downsampler_config
            self._downsampling_period = downsampler_config["downsampling_period"]
            self.offline_dataset_path = training_info.offline_dataset_path
        else:
            self._downsampling_mode = DownsamplingMode.BATCH_THEN_SAMPLE

    def _instantiate_downsampler(
        self, strategy_name: str, downsampler_config: dict, modyn_config: dict, per_sample_loss: torch.nn.modules.loss
    ) -> AbstractRemoteDownsamplingStrategy:
        return instantiate_class(
            "modyn.trainer_server.internal.trainer.remote_downsamplers",
            strategy_name,
            self.pipeline_id,
            self.trigger_id,
            self._batch_size,
            downsampler_config,
            modyn_config,
            per_sample_loss,
            self._device,
            generative=True if self.training_type != "labeled" else False,
        )

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

            elif optimizer_config["source"] == "HuggingFace":
                optimizer_func = getattr(transformers, optimizer_config["algorithm"])
            elif optimizer_config["source"] == "Custom":
                optimizer_module = dynamic_module_import("modyn.trainer_server.custom_optimizers")
                optimizer_func = getattr(optimizer_module, optimizer_config["algorithm"])
            else:
                raise ValueError(
                    f"Unsupported optimizer from {optimizer_config['source']}. PyTorch and APEX are supported"
                )
            optimizer_config_list = []
            for param_group in optimizer_config["param_groups"]:
                module = param_group["module"]

                if optimizer_config["algorithm"] == "Adafactor":  # Check if optimizer is Adafactor
                    no_decay = ["bias", "LayerNorm.weight"]

                    # Create separate parameter group dictionaries
                    param_group_no_decay = copy.deepcopy(param_group["config"])
                    param_group_decay = copy.deepcopy(param_group["config"])

                    param_group_decay["params"] = [
                        p
                        for n, p in eval(f"self._model.{module}.named_parameters()")  # pylint: disable=eval-used
                        if p.requires_grad and not any(m in n for m in no_decay)
                    ]
                    param_group_decay["weight_decay"] = 0.01
                    optimizer_config_list.append(param_group_decay)

                    param_group_no_decay["params"] = [
                        p
                        for n, p in eval(f"self._model.{module}.named_parameters()")  # pylint: disable=eval-used
                        if p.requires_grad and any(m in n for m in no_decay)
                    ]
                    param_group_no_decay["weight_decay"] = 0.0
                    optimizer_config_list.append(param_group_no_decay)

                else:
                    param_group["config"]["params"] = [
                        p
                        for p in eval(f"self._model.{module}.parameters()")  # pylint: disable=eval-used
                        if p.requires_grad
                    ]
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

    def _setup_lr_scheduler(self, training_info: TrainingInfo) -> None:
        self._lr_scheduler = None
        if training_info.lr_scheduler:
            self._step_lr_every = training_info.lr_scheduler["step_every"]

            config_dict = self._update_lr_config_dict(training_info.lr_scheduler["config"])

            if training_info.lr_scheduler["source"] == "Custom":
                lr_scheduler_module = dynamic_module_import("modyn.trainer_server.custom_lr_schedulers")
                custom_lr_scheduler = getattr(lr_scheduler_module, training_info.lr_scheduler["name"])
                optimizers = [self._optimizers[opt] for opt in training_info.lr_scheduler["optimizers"]]
                self._lr_scheduler = custom_lr_scheduler(optimizers, config_dict)
            elif training_info.lr_scheduler["source"] == "PyTorch":
                torch_lr_scheduler = getattr(torch.optim.lr_scheduler, training_info.lr_scheduler["name"])
                if len(training_info.lr_scheduler["optimizers"]) > 1:
                    self._warning("Provided a LR scheduler from PyTorch, but multiple optimizers")
                self._lr_scheduler = torch_lr_scheduler(
                    self._optimizers[training_info.lr_scheduler["optimizers"][0]],
                    **config_dict,
                )
            else:
                raise ValueError(
                    f"Unsupported LR scheduler of source {training_info.lr_scheduler['source']}."
                    "PyTorch and Custom are supported"
                )

    def _seed_trainer_server(self, seed: int) -> None:
        if not (0 <= seed <= 100 and isinstance(seed, int)):
            raise ValueError("The seed must be an integer in the range [0,100]")
        # seed the trainer server
        seed_everything(seed)

    def _calc_expected_sizes(self, downsampling_enabled: bool) -> None:
        num_samples_in_trigger = self.get_num_samples_in_trigger()
        num_samples_per_worker = num_samples_in_trigger // max(self._num_dataloaders, 1)
        batches_per_worker = num_samples_per_worker // self._batch_size

        batches_per_epoch = batches_per_worker * self._num_dataloaders  # We reuse this later
        self._expected_num_batches = batches_per_epoch

        num_samples_per_epoch = (
            self._expected_num_batches * self._batch_size
        )  # scale up again to multiples of batch size

        if downsampling_enabled:
            num_samples_per_epoch = max(
                (self._downsampler.downsampling_ratio * num_samples_per_epoch) // self._downsampling_ratio_max, 1
            )

        self._expected_num_batches = (num_samples_per_epoch // self._batch_size) * self.epochs_per_trigger
        self._expected_num_epochs = self.epochs_per_trigger
        # Handle special case of num_samples_to_pass instead of specifying number of epochs
        if self.num_samples_to_pass > 0:
            self._expected_num_batches = math.ceil(self.num_samples_to_pass / self._batch_size)
            self._expected_num_epochs = math.ceil(self._expected_num_batches / batches_per_epoch)

    # --------------------------------------------------- Sampling --------------------------------------------------- #

    def _sample_then_batch_this_epoch(self, epoch: int) -> bool:
        """Checks if the current epoch should downsample the dataset in
        SAMPLE_THEN_BATCH mode."""
        if self._downsampling_mode != DownsamplingMode.SAMPLE_THEN_BATCH:
            return False

        # self._downsampling_period = 0 : downsample one time per trigger
        if self._downsampling_period == 0:
            return epoch == 0
        # otherwise downsample every self._downsampling_period epochs
        return epoch % self._downsampling_period == 0

    def _iterate_dataloader_and_compute_scores(
        self,
        dataloader: torch.utils.data.DataLoader,
        previous_batch_number: int = -1,
        previous_number_of_samples: int = 0,
    ) -> tuple[int, int]:
        """
        Function to iterate a dataloader, compute the forward pass and send the forward output to the downsampler.
        Args:
            dataloader: torch.dataloader to get the data
            previous_batch_number: The batch number returned from the last call to this method. Useful when this
            function is called several times to keep track of previous invocations (ex label by label dataloader). We
            need to have a total to correctly update the queue and show the progress in the supervisor counter.
            previous_number_of_samples: number of samples processed before calling this function. See above for the use.

        Returns:
            Updated number of batches and samples
        """
        number_of_samples = previous_number_of_samples
        batch_number = previous_batch_number
        for batch in dataloader:
            self.update_queue("DOWNSAMPLING", batch_number, number_of_samples, training_active=False)
            batch_number += 1
            sample_ids, target, data = self.preprocess_batch(batch)
            # Handle cases where target is None for generative tasks
            if self.training_type == "pretraining" and target is None:
                target = torch.Tensor()
            number_of_samples += len(sample_ids)

            no_grad_mgr = torch.no_grad() if isinstance(self._model, DLRM) or isinstance(self._model,T5) else torch.inference_mode()
            context_manager = contextlib.nullcontext() if self._downsampler.requires_grad else no_grad_mgr
            with context_manager:
                with torch.autocast(self._device_type, enabled=self._amp):
                    # compute the scores and accumulate them
                    if(isinstance(self._model,T5) ):
                        model_output=self._model.model(data,labels=target)
                    else:
                        model_output = self._model.model(data) if self._downsampler.forward_required else torch.Tensor()
                    embeddings = self.get_embeddings_if_recorded()
                    self._downsampler.inform_samples(sample_ids, data, model_output, target, embeddings)

        return batch_number, number_of_samples

    # ---------------------------------------------------- Logging --------------------------------------------------- #

    def _info(self, msg: str) -> None:
        self.logger.info(f"[Training {self.training_id}][PL {self.pipeline_id}] {msg}")

    def _warning(self, msg: str) -> None:
        self.logger.warning(f"[Training {self.training_id}][PL {self.pipeline_id}] {msg}")

    def _error(self, msg: str) -> None:
        self.logger.error(f"[Training {self.training_id}][PL {self.pipeline_id}] {msg}")

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
            f"expected size: {expected_size} actual size: "
            + f"{data.shape[0] if isinstance(data, torch.Tensor) else 'n/a'}"
        )
        assert len(sample_ids) == expected_size, f"expected size: {expected_size} actual size: {len(sample_ids)}"
        assert target.shape[0] == expected_size, f"expected size: {expected_size} actual size: {target.shape[0]}"

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

    def end_of_trigger_cleaning(self) -> None:
        self._train_dataloader.dataset.end_of_trigger_cleaning()


def train(
    modyn_config: dict,
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
            modyn_config,
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
        pretrained_path = training_info.pretrained_model_path
        if pretrained_path is not None and pretrained_path.exists():
            pretrained_path.unlink()
