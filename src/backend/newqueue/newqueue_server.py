from concurrent import futures
import os
import sys
from pathlib import Path
from multiprocessing import Process
import time 

import grpc

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from backend.newqueue.newqueue_pb2 import GetNextRequest, GetNextResponse, AddRequest
from backend.newqueue.newqueue_pb2_grpc import NewQueueServicer, add_NewQueueServicer_to_server
from backend.newqueue.newqueue import NewQueue
from storage.storage_pb2_grpc import StorageStub
from storage.storage_pb2 import QueryRequest, QueryResponse


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


def poll(config_dict: dict):
    print("Polling for new data")
    while True:
        time.sleep(config_dict['newqueue']['polling_interval'])
        channel = grpc.insecure_channel(config_dict['storage']['hostname'] +
                                        ':' +
                                        config_dict['storage']['port'])
        stub = StorageStub(channel)
        query_result: QueryResponse = stub.Query(QueryRequest())
        print("Got new data")

        newqueue_channel = grpc.insecure_channel(config_dict['newqueue']['hostname'] +
                                        ':' +
                                        config_dict['newqueue']['port'])
        newqueue_stub = NewQueueServicer(newqueue_channel)
        newqueue_stub.Add(AddRequest(keys=query_result.keys))

def serve(config_dict: dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    add_NewQueueServicer_to_server(
        NewQueueServicer(config_dict), server)
    print(
        'Starting server. Listening on port .' +
        config_dict['newqueue']['port'])
    server.add_insecure_port('[::]:' + config_dict['newqueue']['port'])
    server.start()

    source_process = Process(target=poll, args=(config_dict,))
    source_process.start()

    server.wait_for_termination()


if __name__ == '__main__':
    import sys
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python newqueue_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
