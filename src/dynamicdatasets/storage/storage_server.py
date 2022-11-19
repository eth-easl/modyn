from concurrent import futures
from threading import Thread

import grpc

from dynamicdatasets.storage.storage_pb2 import GetRequest, GetResponse, PutRequest, PutResponse
from dynamicdatasets.storage.storage_pb2_grpc import StorageServicer, add_StorageServicer_to_server


class StorageServicer(StorageServicer):
    """Provides methods that implement functionality of the storage server."""

    def __init__(self, config: dict):
        super().__init__()

        adapter_module = self.my_import('dynamicdatasets.storage.adapter')
        self.__adapter = getattr(
            adapter_module,
            config['storage']['adapter'])(config)

    def Get(self, request: GetRequest, context):
        print("Getting data")
        data = self.__adapter.get(request.keys)
        if data is None:
            return GetResponse(dataMap={})
        else:
            return GetResponse(dataMap=data)

    def Put(self, request: PutRequest, context):
        print("Putting data")
        self.__adapter.put(request.keys, request.value)
        return PutResponse()

    def my_import(self, name):
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod


def serve(config_dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_StorageServicer_to_server(
        StorageServicer(config_dict), server)
    print(
        'Starting server. Listening on port .' +
        config_dict['storage']['port'])
    server.add_insecure_port('[::]:' + config_dict['storage']['port'])
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
