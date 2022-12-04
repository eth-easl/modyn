from concurrent import futures
from multiprocessing import Process
import os
import sys

import grpc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from storage.storage_pb2 import GetRequest, GetResponse, PutRequest, PutResponse, QueryRequest, QueryResponse
from storage.storage_pb2_grpc import StorageServicer, add_StorageServicer_to_server


class StorageServicer(StorageServicer):
    """Provides methods that implement functionality of the storage server."""

    def __init__(self, config: dict):
        super().__init__()

        adapter_module = my_import('storage.adapter')
        self.__adapter = getattr(
            adapter_module,
            config['storage']['adapter'])(config)

    def Query(self, request: QueryRequest, context):
        print("Query for data")
        keys = self.__adapter.query()
        return QueryResponse(keys=keys)

    def Get(self, request: GetRequest, context):
        print("Getting data")
        data = self.__adapter.get(request.keys)
        return GetResponse(value=data)

    def Put(self, request: PutRequest, context):
        print("Putting data")
        self.__adapter.put(request.keys, request.value)
        return PutResponse()

def my_import(name):
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

    if (config['storage']['data_source']['enabled']):
        source_module = my_import('storage.datasource')
        source = getattr(
            source_module,
            config['storage']['data_source']['type'])(config)
        source_process = Process(target=source.run, args=())
        source_process.start()

    server.wait_for_termination()
    source_process.join()


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
