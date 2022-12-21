import grpc
import os
import sys
from concurrent import futures
from pathlib import Path
import logging

import yaml

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from modyn.frontend.dynamicdatasets.trainer.trainer_server_pb2_grpc import TrainerServerServicer, add_TrainerServerServicer_to_server
from modyn.frontend.dynamicdatasets.trainer.trainer_server_pb2 import TrainerServerRequest, TrainerServerResponse

logging.basicConfig(format='%(asctime)s %(message)s')

class TrainerGRPCServer:
    """Implements necessary functionality in order to communicate with the supervisor."""

    def __init__(self):
        pass

    def start_training(self, request: TrainerServerRequest, context: grpc.ServicerContext) -> TrainerServerResponse:
        return TrainerServerResponse(training_id=10)

def serve(config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_TrainerServerServicer_to_server(
        TrainerGRPCServer(), server)
    logging.info(
        'Starting trainer server. Listening on port .' +
        config['trainer']['port'])
    server.add_insecure_port('[::]:' + config['trainer']['port'])
    print("start serving!")
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python trainer_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
