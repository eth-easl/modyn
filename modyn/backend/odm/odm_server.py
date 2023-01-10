import logging
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc
import yaml
from backend.odm.odm import OptimalDatasetMetadata
from backend.odm.odm_pb2_grpc import ODMServicer, add_ODMServicer_to_server

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from backend.odm.odm_pb2 import (  # noqa: E501, E402
    DeleteRequest,
    DeleteResponse,
    GetByKeysRequest,
    GetByQueryRequest,
    GetKeysResponse,
    GetResponse,
    SetRequest,
    SetResponse,
)

logging.basicConfig(format="%(asctime)s %(message)s")


class ODMGRPCServer(ODMServicer):
    """Provides methods that implement functionality of the ODM server."""

    def __init__(self, config: dict):
        super().__init__()
        self.__config = config
        self.__odm = OptimalDatasetMetadata(config)

    def GetByKeys(self, request: GetByKeysRequest, context: grpc.ServicerContext) -> GetResponse:
        logging.info("Getting data by keys")
        keys, score, data = self.__odm.get_by_keys(request.keys, request.training_id)
        return GetResponse(keys=keys, data=data, scores=score)

    def GetByQuery(self, request: GetByQueryRequest, context: grpc.ServicerContext) -> GetResponse:
        logging.info("Getting data by query")
        keys, score, data = self.__odm.get_by_query(request.keys)
        return GetResponse(keys=keys, data=data, scores=score)

    def GetKeysByQuery(self, request: GetByQueryRequest, context: grpc.ServicerContext) -> GetKeysResponse:
        logging.info("Getting keys by query")
        keys = self.__odm.get_keys_by_query(request.keys)
        return GetKeysResponse(keys=keys)

    def Set(self, request: SetRequest, context: grpc.ServicerContext) -> SetResponse:
        logging.info("Setting data")
        self.__odm.set(request.keys, request.scores, request.data, request.training_id)
        return SetResponse()

    def DeleteTraining(self, request: DeleteRequest, context: grpc.ServicerContext) -> DeleteResponse:
        logging.info("Deleting training data")
        self.__odm.delete_training(request.training_id)
        return DeleteResponse()


def serve(config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ODMServicer_to_server(ODMGRPCServer(config), server)
    logging.info("Starting server. Listening on port ." + config["odm"]["port"])
    server.add_insecure_port(f'[::]:{config["odm"]["port"]}')
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python odm_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
