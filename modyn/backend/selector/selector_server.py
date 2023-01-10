import logging
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc
import yaml

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from backend.selector.new_data_selector import NewDataSelector  # noqa: E402
from backend.selector.selector_pb2 import (  # noqa: E402; noqa: E402, E501
    GetSamplesRequest,
    RegisterTrainingRequest,
    SamplesResponse,
    TrainingResponse,
)
from backend.selector.selector_pb2_grpc import SelectorServicer, add_SelectorServicer_to_server  # noqa: E402

logging.basicConfig(format="%(asctime)s %(message)s")


class SelectorGRPCServer(SelectorServicer):
    """Provides methods that implement functionality of the metadata server."""

    def __init__(self, config: dict):
        # selector_module = dynamic_module_import('dynamicdatasets.selector')
        # self._selector = getattr(selector_module,config['metadata']['selector'])(config)
        self._selector = NewDataSelector(config)

    def register_training(self, request: RegisterTrainingRequest, context: grpc.ServicerContext) -> TrainingResponse:
        logging.info("Registering training with request - " + str(request))
        training_id = self._selector.register_training(request.training_set_size, request.num_workers)
        return TrainingResponse(training_id=training_id)

    def get_sample_keys(self, request: GetSamplesRequest, context: grpc.ServicerContext) -> SamplesResponse:
        logging.info("Fetching samples for request - " + str(request))
        samples_keys = self._selector.get_sample_keys(
            request.training_id, request.training_set_number, request.worker_id
        )
        return SamplesResponse(training_samples_subset=samples_keys)


def serve(config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_SelectorServicer_to_server(SelectorGRPCServer(config), server)
    logging.info("Starting server. Listening on port ." + config["selector"]["port"])
    server.add_insecure_port("[::]:" + config["selector"]["port"])
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python selector_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
