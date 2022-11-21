from concurrent import futures
from threading import Thread

import grpc
import persistqueue

from dynamicdatasets.newqueue.newqueue_pb2 import AddRequest, AddResponse, GetNextRequest, GetNextResponse
from dynamicdatasets.newqueue.newqueue_pb2_grpc import NewQueueServicer, add_NewQueueServicer_to_server


class NewQueueServicer(NewQueueServicer):
    """Provides methods that implement functionality of the new queue server."""

    def __init__(self, config: dict):
        super().__init__()
        self.__queue = persistqueue.Queue(config['newqueue']['path'])

    def Add(self, request: AddRequest, context) -> AddResponse:
        print("Adding data")
        for key in request.keys:
            self.__queue.put(key)
        return AddResponse()

    def GetNext(self, request: GetNextRequest, context) -> GetNextResponse:
        print("Getting next data")
        keys = []
        for i in range(request.limit):
            keys.append(self.__queue.get())
        return GetNextResponse(keys=keys)


def serve(config_dict):
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
    import logging
    import sys
    import yaml

    logging.basicConfig()
    with open(sys.argv[1], 'r') as stream:
        config = yaml.safe_load(stream)
    serve(config)
