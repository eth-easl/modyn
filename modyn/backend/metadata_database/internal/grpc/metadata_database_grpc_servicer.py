# pylint: disable-next=no-name-in-module
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import GetByKeysRequest, GetByQueryRequest, GetResponse, SetRequest, SetResponse, GetKeysResponse, DeleteRequest, DeleteResponse, RegisterRequest, RegisterResponse, GetTrainingRequest, TrainingResponse  # noqa: E501, E402
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataServicer, add_MetadataServicer_to_server  # noqa: E501, E402
from modyn.backend.metadata_database.metadata_database import MetadataDatabase
from concurrent import futures
import os
import sys
from pathlib import Path
import logging

import grpc

logger = logging.getLogger(__name__)

path = Path(os.path.abspath(__file__))
SCRIPT_DIR = path.parent.parent.absolute()
sys.path.append(os.path.dirname(SCRIPT_DIR))


class MetadataDatabaseGRPCServicer(MetadataServicer):
    """Provides methods that implement functionality of the metadata database server."""

    def __init__(self, config: dict):
        super().__init__()
        self.__config = config
        self.__metadata_database = MetadataDatabase(self.__config)

    def GetByKeys(self, request: GetByKeysRequest, context: grpc.ServicerContext) -> GetResponse:
        logger.info("Getting data by keys")
        keys, score, seen, labels, data = self.__metadata_database.get_by_keys(
            request.keys, request.training_id)
        return GetResponse(keys=keys, data=data, scores=score, seen=seen, label=labels)

    def GetByQuery(self, request: GetByQueryRequest, context: grpc.ServicerContext) -> GetResponse:
        logger.info("Getting data by query")
        keys, score, seen, labels, data = self.__metadata_database.get_by_query(request.query)
        return GetResponse(keys=keys, data=data, scores=score, seen=seen, label=labels)

    def GetKeysByQuery(self, request: GetByQueryRequest, context: grpc.ServicerContext) -> GetKeysResponse:
        logger.info("Getting keys by query")
        keys = self.__metadata_database.get_keys_by_query(request.query)
        return GetKeysResponse(keys=keys)

    def Set(self, request: SetRequest, context: grpc.ServicerContext) -> SetResponse:
        logger.info("Setting data")
        self.__metadata_database.set(
            request.keys,
            request.scores,
            request.seen,
            request.label,
            request.data,
            request.training_id)
        return SetResponse()

    def DeleteTraining(self, request: DeleteRequest, context: grpc.ServicerContext) -> DeleteResponse:
        logger.info("Deleting training data")
        self.__metadata_database.delete_training(request.training_id)
        return DeleteResponse()

    def RegisterTraining(self, request: RegisterRequest, context: grpc.ServicerContext) -> RegisterResponse:
        training_id = self.__metadata_database.register_training(request.training_set_size, request.num_workers)
        logger.info(f'Registered training {training_id}')
        return RegisterResponse(training_id=training_id)

    def GetTrainingInfo(self, request: GetTrainingRequest, context: grpc.ServicerContext) -> TrainingResponse:
        logger.info(f'Getting training info for {request.training_id}')
        training_set_size, num_workers = self.__metadata_database.get_training_info(request.training_id)
        return TrainingResponse(training_set_size=training_set_size, num_workers=num_workers)


def serve(config: dict) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_MetadataServicer_to_server(MetadataDatabaseGRPCServicer(config), server)
    logger.info(f'Starting server. Listening on port {config["metadata_database"]["port"]}.')
    server.add_insecure_port(f'[::]:{config["metadata_database"]["port"]}')
    server.start()
    server.wait_for_termination()
