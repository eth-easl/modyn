from concurrent import futures
import os
import sys
from pathlib import Path

import grpc

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from backend.newqueue.newqueue_pb2 import GetNextRequest, GetNextResponse, AddRequest, AddResponse
from backend.newqueue.newqueue_pb2_grpc import NewQueueServicer, add_NewQueueServicer_to_server
from backend.newqueue.newqueue import NewQueue


class NewQueueServicer(NewQueueServicer):
    """Provides methods that implement functionality of the new queue server."""

    def __init__(self, config: dict):
        super().__init__()
        self.__config = config
        self.__queue = NewQueue(config)

    def GetNext(self, request: GetNextRequest, context) -> GetNextResponse:
        print("Getting next data")
        keys = []
        keys = self.__queue.get_next(request.limit, request.training_id)
        return GetNextResponse(keys=keys)

    def Add(self, request: AddRequest, context) -> AddResponse:
        print("Adding data")
        self.__queue.add(request.keys)
        return AddResponse()


def serve(config_dict: dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_NewQueueServicer_to_server(
        NewQueueServicer(config_dict), server)
    print(
        'Starting server. Listening on port .' +
        config_dict['newqueue']['port'])
    server.add_insecure_port('[::]:' + config_dict['newqueue']['port'])
    server.start()

    server.wait_for_termination()


if __name__ == '__main__':
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python newqueue_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
