import logging
import os
import sys
from pathlib import Path

import grpc

# Pylint cannot handle the auto-generated gRPC files, apparently.
# pylint: disable-next=no-name-in-module
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import (  # noqa: E402, E501
    GetSamplesRequest,
    RegisterTrainingRequest,
    SamplesResponse,
    TrainingResponse,
)
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import (
    SelectorServicer,
)  # noqa: E402, E501
from modyn.backend.selector.selector_strategy import SelectorStrategy

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


logger = logging.getLogger(__name__)


class SelectorGRPCServicer(SelectorServicer):
    """Provides methods that implement functionality of the metadata server."""

    def __init__(self, strategy: SelectorStrategy):
        self.selector_strategy = strategy

    def register_training(self, request: RegisterTrainingRequest, context: grpc.ServicerContext) -> TrainingResponse:
        logger.info(f"Registering training with request - {str(request)}")
        training_id = self.selector_strategy.register_training(request.training_set_size, request.num_workers)
        return TrainingResponse(training_id=training_id)

    def get_sample_keys(self, request: GetSamplesRequest, context: grpc.ServicerContext) -> SamplesResponse:
        logger.info(f"Fetching samples for request - {str(request)}")
        samples_keys = self.selector_strategy.get_sample_keys(
            request.training_id, request.training_set_number, request.worker_id
        )
        samples_keys = [sample[0] for sample in samples_keys]
        return SamplesResponse(training_samples_subset=samples_keys)
