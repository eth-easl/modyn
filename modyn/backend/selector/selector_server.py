import grpc
import os
import sys
from concurrent import futures
from pathlib import Path
import logging

import yaml

from modyn.utils import dynamic_module_import

from backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorServicer, add_SelectorServicer_to_server  # noqa: E402, E501
# Pylint cannot handle the auto-generated gRPC files, apparently.
# pylint: disable-next=no-name-in-module
from backend.selector.internal.grpc.generated.selector_pb2 import RegisterTrainingRequest, GetSamplesRequest, SamplesResponse, TrainingResponse  # noqa: E402, E501

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


logging.basicConfig(format='%(asctime)s %(message)s')


class SelectorGRPCServer(SelectorServicer):
    """Provides methods that implement functionality of the metadata server."""

    def __init__(self, config: dict):
        selector_module = dynamic_module_import(
            f"modyn.backend.selector.custom_selectors.{config['selector']['package']}")
        self._selector = getattr(selector_module, config['selector']['class'])(config)

    def register_training(self, request: RegisterTrainingRequest, context: grpc.ServicerContext) -> TrainingResponse:
        logging.info(f"Registering training with request - {str(request)}")
        training_id = self._selector.register_training(
            request.training_set_size, request.num_workers)
        return TrainingResponse(training_id=training_id)

    def get_sample_keys(self, request: GetSamplesRequest, context: grpc.ServicerContext) -> SamplesResponse:
        logging.info(f"Fetching samples for request - {str(request)}")
        samples_keys = self._selector.get_sample_keys(
            request.training_id, request.training_set_number, request.worker_id)
        samples_keys = [sample[0] for sample in samples_keys]
        return SamplesResponse(training_samples_subset=samples_keys)


def serve(config: dict, servicer: SelectorGRPCServer) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_SelectorServicer_to_server(servicer, server)
    logging.info(f"Starting server. Listening on port {config['selector']['port']}.")
    server.add_insecure_port('[::]:' + config['selector']['port'])
    server.start()
    server.wait_for_termination()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python selector_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    serve(config, SelectorGRPCServer(config))


if __name__ == '__main__':
    main()
