from utils import dynamic_module_import
from concurrent import futures
import os
import sys
import logging

import grpc
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from storage.storage_pb2 import GetRequest, GetResponse, PutRequest, PutResponse, QueryRequest, QueryResponse  # noqa: E501, E402
from storage.storage_pb2_grpc import StorageServicer, add_StorageServicer_to_server  # noqa: E402


logging.basicConfig(format='%(asctime)s %(message)s')


class StorageServicer(StorageServicer):
    """Provides methods that implement functionality of the storage server."""

    def __init__(self, config: dict):
        super().__init__()

        adapter_module = dynamic_module_import('storage.adapter')
        self.__adapter = getattr(
            adapter_module,
            config['storage']['adapter'])(config)

    def Query(self, request: QueryRequest, context):
        logging.info("Storage: Query for data")
        keys = self.__adapter.query()
        return QueryResponse(keys=keys)

    def Get(self, request: GetRequest, context):
        logging.info("Storage: Getting data")
        data = self.__adapter.get(request.keys)
        return GetResponse(value=data)

    def Put(self, request: PutRequest, context):
        logging.info("Storage: Putting data")
        self.__adapter.put(request.keys, request.value)
        return PutResponse()


def serve(config_dict):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_StorageServicer_to_server(
        StorageServicer(config_dict), server)
    logging.info(
        'Starting server. Listening on port .' +
        config_dict['storage']['port'])
    server.add_insecure_port('[::]:' + config_dict['storage']['port'])
    server.start()

    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python storage_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
