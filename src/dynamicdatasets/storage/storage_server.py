from concurrent import futures

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
        return GetResponse(value=data)

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
    import sys
    import yaml
    import logging

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python storage_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
