from modyn.backend.metadata_database.metadata_pb2_grpc import MetadataServicer, add_MetadataServicer_to_server
from modyn.backend.metadata_database.metadata_database import MetadataDatabase
from concurrent import futures
import os
import sys
from pathlib import Path
import logging

import grpc
import yaml

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))

from backend.metadata_database.metadata_pb2 import GetByKeysRequest, GetByQueryRequest, GetResponse, SetRequest, SetResponse, GetKeysResponse, DeleteRequest, DeleteResponse, RegisterRequest, RegisterResponse, GetTrainingRequest, TrainingResponse  # noqa: E501, E402

logging.basicConfig(format='%(asctime)s %(message)s')


class MetadataDatabaseGRPCServer(MetadataServicer):
    """Provides methods that implement functionality of the metadata database server."""

    def __init__(self, config: dict):
        super().__init__()
        self.__config = config
        self.__metadata_database = MetadataDatabase(config)

    def GetByKeys(self, request: GetByKeysRequest, context: grpc.ServicerContext) -> GetResponse:
        logging.info("Getting data by keys")
        keys, score, data = self.__metadata_database.get_by_keys(
            request.keys, request.training_id)
        return GetResponse(keys=keys, data=data, scores=score)

    def GetByQuery(self, request: GetByQueryRequest, context: grpc.ServicerContext) -> GetResponse:
        logging.info("Getting data by query")
        keys, score, data = self.__metadata_database.get_by_query(request.keys)
        return GetResponse(keys=keys, data=data, scores=score)

    def GetKeysByQuery(self, request: GetByQueryRequest, context: grpc.ServicerContext) -> GetKeysResponse:
        logging.info("Getting keys by query")
        keys = self.__metadata_database.get_keys_by_query(request.keys)
        return GetKeysResponse(keys=keys)

    def Set(self, request: SetRequest, context: grpc.ServicerContext) -> SetResponse:
        logging.info("Setting data")
        self.__metadata_database.set(
            request.keys,
            request.scores,
            request.data,
            request.training_id)
        return SetResponse()

    def DeleteTraining(self, request: DeleteRequest, context: grpc.ServicerContext) -> DeleteResponse:
        logging.info("Deleting training data")
        self.__metadata_database.delete_training(request.training_id)
        return DeleteResponse()

    def RegisterTraining(self, request: RegisterRequest, context: grpc.ServicerContext) -> RegisterResponse:
        logging.info(f'Registering training {request.training_id}')
        self.__metadata_database.register_training(request.training_id, request.training_set_size, request.num_workers)
        return RegisterResponse()

    def GetTrainingInfo(self, request: GetTrainingRequest, context: grpc.ServicerContext) -> TrainingResponse:
        logging.info(f'Getting training info for {request.training_id}')
        training_set_size, num_workers = self.__metadata_database.get_training_info(request.training_id)
        return TrainingResponse(training_set_size=training_set_size, num_workers=num_workers)


def serve(config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MetadataServicer_to_server(MetadataDatabaseGRPCServer(config), server)
    logging.info(
        'Starting server. Listening on port .' +
        config["metadata_database"]["port"])
    server.add_insecure_port(f'[::]:{config["metadata_database"]["port"]}')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python metadata_database_server.py <config_file>")
        sys.exit(1)

    with open(sys.argv[1], "r") as f:
        config = yaml.safe_load(f)

    serve(config)
