import json
import logging
import os
import threading
from collections.abc import Iterable

import grpc

# pylint: disable=no-name-in-module
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    AvailableLabelsResponse,
    DataInformRequest,
    DataInformResponse,
    GetAvailableLabelsRequest,
    GetNumberOfPartitionsRequest,
    GetNumberOfSamplesRequest,
    GetSamplesRequest,
    GetSelectionStrategyRequest,
    GetStatusBarScaleRequest,
    JsonString,
    NumberOfPartitionsResponse,
    NumberOfSamplesResponse,
    SamplesResponse,
    SeedSelectorRequest,
    SeedSelectorResponse,
    SelectionStrategyResponse,
    StatusBarScaleResponse,
    TriggerResponse,
    UsesWeightsRequest,
    UsesWeightsResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorServicer  # noqa: E402, E501
from modyn.selector.internal.selector_manager import SelectorManager
from modyn.utils import seed_everything

logger = logging.getLogger(__name__)

# TODO(#124): Add a function to unregister a pipeline


class SelectorGRPCServicer(SelectorServicer):
    """Provides methods that implement functionality of the selector."""

    def __init__(self, selector_manager: SelectorManager, sample_batch_size: int):
        self.selector_manager = selector_manager
        self._sample_batch_size = sample_batch_size

    def get_sample_keys_and_weights(  # pylint: disable-next=unused-argument, too-many-locals
        self, request: GetSamplesRequest, context: grpc.ServicerContext
    ) -> Iterable[SamplesResponse]:
        pipeline_id, trigger_id, worker_id, partition_id = (
            request.pipeline_id,
            request.trigger_id,
            request.worker_id,
            request.partition_id,
        )
        tid = threading.get_native_id()
        pid = os.getpid()

        _logmsg = (
            f"[{pid}][{tid}][Pipeline {pipeline_id}]: Fetching samples for trigger id {trigger_id}"
            + f" and worker id {worker_id} and partition id {partition_id}"
        )
        logger.info(_logmsg)

        samples = self.selector_manager.get_sample_keys_and_weights(pipeline_id, trigger_id, worker_id, partition_id)

        num_samples = len(samples)
        if num_samples == 0:
            logger.info("No samples found.")
            yield SamplesResponse()
            return

        for i in range(0, num_samples, self._sample_batch_size):
            batch = samples[i : i + self._sample_batch_size]
            batch_keys = [sample[0] for sample in batch]
            batch_weights = [sample[1] for sample in batch]
            yield SamplesResponse(training_samples_subset=batch_keys, training_samples_weights=batch_weights)

    def inform_data(self, request: DataInformRequest, context: grpc.ServicerContext) -> DataInformResponse:
        pipeline_id, keys, timestamps, labels = request.pipeline_id, request.keys, request.timestamps, request.labels
        tid = threading.get_native_id()
        pid = os.getpid()
        logger.info(f"[{pid}][{tid}][Pipeline {pipeline_id}]: Selector is informed of {len(keys)} new data points")

        log = self.selector_manager.inform_data(pipeline_id, keys, timestamps, labels)
        return DataInformResponse(log=JsonString(value=json.dumps(log)))

    def inform_data_and_trigger(self, request: DataInformRequest, context: grpc.ServicerContext) -> TriggerResponse:
        pipeline_id, keys, timestamps, labels = request.pipeline_id, request.keys, request.timestamps, request.labels
        tid = threading.get_native_id()
        pid = os.getpid()
        _lgmsg = (
            f"[{pid}][{tid}][Pipeline {pipeline_id}]: Selector is informed of {len(keys)} new data points"
            + f"+ trigger at timestamp {timestamps[-1] if len(keys) > 0 else 'n/a'}"
        )
        logger.info(_lgmsg)

        trigger_id, log = self.selector_manager.inform_data_and_trigger(pipeline_id, keys, timestamps, labels)
        return TriggerResponse(trigger_id=trigger_id, log=JsonString(value=json.dumps(log)))

    def get_number_of_samples(  # pylint: disable-next=unused-argument
        self, request: GetNumberOfSamplesRequest, context: grpc.ServicerContext
    ) -> NumberOfSamplesResponse:
        pipeline_id, trigger_id = request.pipeline_id, request.trigger_id
        logger.info(f"[Pipeline {pipeline_id}]: Received number of samples request for trigger id {trigger_id}")

        num_samples = self.selector_manager.get_number_of_samples(pipeline_id, trigger_id)

        return NumberOfSamplesResponse(num_samples=num_samples)

    def get_status_bar_scale(  # pylint: disable-next=unused-argument
        self, request: GetStatusBarScaleRequest, context: grpc.ServicerContext
    ) -> StatusBarScaleResponse:
        pipeline_id = request.pipeline_id
        logger.info(f"[Pipeline {pipeline_id}]: Received status bar scale request")

        status_bar_scale = self.selector_manager.get_status_bar_scale(pipeline_id)

        return StatusBarScaleResponse(status_bar_scale=status_bar_scale)

    def get_number_of_partitions(  # pylint: disable-next=unused-argument
        self, request: GetNumberOfPartitionsRequest, context: grpc.ServicerContext
    ) -> NumberOfPartitionsResponse:
        pipeline_id, trigger_id = request.pipeline_id, request.trigger_id
        logger.info(f"[Pipeline {pipeline_id}]: Received number of partitions request for trigger id {trigger_id}")

        num_partitions = self.selector_manager.get_number_of_partitions(pipeline_id, trigger_id)

        return NumberOfPartitionsResponse(num_partitions=num_partitions)

    def get_available_labels(  # pylint: disable-next=unused-argument
        self, request: GetAvailableLabelsRequest, context: grpc.ServicerContext
    ) -> AvailableLabelsResponse:
        pipeline_id = request.pipeline_id
        logger.info(f"[Pipeline {pipeline_id}]: Received get available labels request")

        available_labels = self.selector_manager.get_available_labels(pipeline_id)

        return AvailableLabelsResponse(available_labels=available_labels)

    def uses_weights(  # pylint: disable-next=unused-argument
        self, request: UsesWeightsRequest, context: grpc.ServicerContext
    ) -> UsesWeightsResponse:
        pipeline_id = request.pipeline_id
        logger.info(f"[Pipeline {pipeline_id}]: Received is weighted request")

        uses_weights = self.selector_manager.uses_weights(pipeline_id)

        return UsesWeightsResponse(uses_weights=uses_weights)

    def get_selection_strategy(  # pylint: disable-next=unused-argument
        self, request: GetSelectionStrategyRequest, context: grpc.ServicerContext
    ) -> SelectionStrategyResponse:
        pipeline_id = request.pipeline_id
        logger.info(f"[Pipeline {pipeline_id}]: Received selection strategy request")

        (
            downsampling_enabled,
            name,
            downsampler_config,
        ) = self.selector_manager.get_selection_strategy_remote(pipeline_id)

        downsampler_config_str = json.dumps(downsampler_config)

        return SelectionStrategyResponse(
            downsampling_enabled=downsampling_enabled,
            strategy_name=name,
            downsampler_config=JsonString(value=downsampler_config_str),
        )

    def seed_selector(  # pylint: disable-next=unused-argument
        self, request: SeedSelectorRequest, context: grpc.ServicerContext
    ) -> SeedSelectorResponse:
        seed = request.seed
        assert 0 <= seed <= 100
        logger.info(f"Received seed request with seed {seed}")

        seed_everything(seed)

        return SeedSelectorResponse(success=True)
