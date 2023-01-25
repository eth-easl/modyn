import logging

import grpc

# pylint: disable-next=no-name-in-module
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (  # noqa: E402, E501
    DataInformRequest,
    Empty,
    GetSamplesRequest,
    PipelineResponse,
    RegisterPipelineRequest,
    SamplesResponse,
    TriggerResponse,
)
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorServicer  # noqa: E402, E501
from modyn.backend.selector.internal.selector_manager import SelectorManager

logger = logging.getLogger(__name__)


class SelectorGRPCServicer(SelectorServicer):
    """Provides methods that implement functionality of the selector."""

    def __init__(self, selector_manager: SelectorManager):
        self.selector_manager = selector_manager

    def register_pipeline(self, request: RegisterPipelineRequest, context: grpc.ServicerContext) -> PipelineResponse:
        logger.info(f"Registering training with request - {str(request)}")
        pipeline_id = self.selector_manager.register_pipeline(request.num_workers, request.selector_strategy_config)
        return PipelineResponse(pipeline_id=pipeline_id)

    def get_sample_keys_and_weight(  # pylint: disable-next=unused-argument
        self, request: GetSamplesRequest, context: grpc.ServicerContext
    ) -> SamplesResponse:
        logger.info(f"Fetching samples for request - {str(request)}")
        samples = self.selector_manager.get_sample_keys_and_weight(
            request.pipeline_id, request.trigger_id, request.worker_id
        )
        samples_keys = [sample[0] for sample in samples]
        samples_weights = [sample[1] for sample in samples]
        return SamplesResponse(training_samples_subset=samples_keys, training_samples_weight=samples_weights)

    def inform_data(self, request: DataInformRequest, context: grpc.ServicerContext) -> Empty:
        pipeline_id, keys, timestamps, labels = request.pipeline_id, request.keys, request.timestamps, request.labels
        logger.info(f"Selctor is informed of {len(keys)} new data points")
        self.selector_manager.inform_data(pipeline_id, keys, timestamps, labels)
        return Empty()

    def inform_data_and_trigger(self, request: DataInformRequest, context: grpc.ServicerContext) -> TriggerResponse:
        pipeline_id, keys, timestamps, labels = request.pipeline_id, request.keys, request.timestamps, request.labels
        logger.info(f"Selctor is informed of {len(keys)} new data points, and triggered")
        trigger_id = self.selector_manager.inform_data_and_trigger(pipeline_id, keys, timestamps, labels)
        return TriggerResponse(trigger_id=trigger_id)
