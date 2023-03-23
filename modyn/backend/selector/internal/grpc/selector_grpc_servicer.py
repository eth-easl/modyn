import json
import logging
from typing import Iterable

import grpc

# pylint: disable=no-name-in-module
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (  # noqa: E402, E501
    DataInformRequest,
    Empty,
    GetNumberOfPartitionsRequest,
    GetNumberOfSamplesRequest,
    GetSamplesRequest,
    GetSelectionStrategyRequest,
    JsonString as SelectorJsonString,
    NumberOfPartitionsResponse,
    NumberOfSamplesResponse,
    PipelineResponse,
    RegisterPipelineRequest,
    SamplesResponse,
    SelectionStrategyResponse,
    TriggerResponse,
)
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorServicer  # noqa: E402, E501
from modyn.backend.selector.internal.selector_manager import SelectorManager

logger = logging.getLogger(__name__)

# TODO(#124): Add a function to unregister a pipeline


class SelectorGRPCServicer(SelectorServicer):
    """Provides methods that implement functionality of the selector."""

    def __init__(self, selector_manager: SelectorManager, sample_batch_size: int):
        self.selector_manager = selector_manager
        self._sample_batch_size = sample_batch_size

    def register_pipeline(self, request: RegisterPipelineRequest, context: grpc.ServicerContext) -> PipelineResponse:
        logger.info(f"Registering pipeline with request - {str(request)}")
        pipeline_id = self.selector_manager.register_pipeline(request.num_workers, request.selection_strategy.value)
        return PipelineResponse(pipeline_id=pipeline_id)

    def get_sample_keys_and_weights(  # pylint: disable-next=unused-argument
        self, request: GetSamplesRequest, context: grpc.ServicerContext
    ) -> Iterable[SamplesResponse]:
        pipeline_id, trigger_id, worker_id, partition_id = (
            request.pipeline_id,
            request.trigger_id,
            request.worker_id,
            request.partition_id,
        )
        logger.info(
            f"[Pipeline {pipeline_id}]: Fetching samples for trigger id {trigger_id}"
            + f" and worker id {worker_id} and partition id {partition_id}"
        )

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

    def inform_data(self, request: DataInformRequest, context: grpc.ServicerContext) -> Empty:
        pipeline_id, keys, timestamps, labels = request.pipeline_id, request.keys, request.timestamps, request.labels
        logger.info(f"[Pipeline {pipeline_id}]: Selector is informed of {len(keys)} new data points")

        self.selector_manager.inform_data(pipeline_id, keys, timestamps, labels)
        return Empty()

    def inform_data_and_trigger(self, request: DataInformRequest, context: grpc.ServicerContext) -> TriggerResponse:
        pipeline_id, keys, timestamps, labels = request.pipeline_id, request.keys, request.timestamps, request.labels
        logger.info(
            f"[Pipeline {pipeline_id}]: Selector is informed of {len(keys)} new data points"
            + f"+ trigger at timestamp {timestamps[-1] if len(keys) > 0 else 'n/a'}"
        )

        trigger_id = self.selector_manager.inform_data_and_trigger(pipeline_id, keys, timestamps, labels)
        return TriggerResponse(trigger_id=trigger_id)

    def get_number_of_samples(  # pylint: disable-next=unused-argument
        self, request: GetNumberOfSamplesRequest, context: grpc.ServicerContext
    ) -> NumberOfSamplesResponse:
        pipeline_id, trigger_id = request.pipeline_id, request.trigger_id
        logger.info(f"[Pipeline {pipeline_id}]: Received number of samples request for trigger id {trigger_id}")

        num_samples = self.selector_manager.get_number_of_samples(pipeline_id, trigger_id)

        return NumberOfSamplesResponse(num_samples=num_samples)

    def get_number_of_partitions(  # pylint: disable-next=unused-argument
        self, request: GetNumberOfPartitionsRequest, context: grpc.ServicerContext
    ) -> NumberOfPartitionsResponse:
        pipeline_id, trigger_id = request.pipeline_id, request.trigger_id
        logger.info(f"[Pipeline {pipeline_id}]: Received number of partitions request for trigger id {trigger_id}")

        num_partitions = self.selector_manager.get_number_of_partitions(pipeline_id, trigger_id)

        return NumberOfPartitionsResponse(num_partitions=num_partitions)

    def get_selection_strategy(  # pylint: disable-next=unused-argument
        self, request: GetSelectionStrategyRequest, context: grpc.ServicerContext
    ) -> SelectionStrategyResponse:
        pipeline_id = request.pipeline_id
        logger.info(f"[Pipeline {pipeline_id}]: Received selection strategy request")

        downsampling_enabled, name, params = self.selector_manager.get_selection_strategy_remote(pipeline_id)

        params = json.dumps(params)

        return SelectionStrategyResponse(
            downsampling_enabled=downsampling_enabled, strategy_name=name, params=SelectorJsonString(value=params)
        )
