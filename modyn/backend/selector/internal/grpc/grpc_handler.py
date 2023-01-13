import logging

import grpc
from modyn import utils

# pylint: disable-next=no-name-in-module
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import (  # noqa: E501, E402
    GetByQueryRequest,
    GetTrainingRequest,
    RegisterRequest,
)
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataStub

TIMEOUT_SEC = 5
logger = logging.getLogger(__name__)


class GRPCHandler:
    def __init__(self, modyn_config: dict):
        self.config = modyn_config
        self.connected_to_metadata = False

        self._init_metadata()

    def _init_metadata(self) -> None:
        assert self.config is not None
        address = f"{self.config['metadata_database']['hostname']}:{self.config['metadata_database']['port']}"
        self.metadata_database_channel = grpc.insecure_channel(address)

        if not utils.grpc_connection_established(self.metadata_database_channel, timeout_sec=TIMEOUT_SEC):
            raise ConnectionError(f"Could not establish gRPC connection to metadata server at {address}.")

        self.metadata_database = MetadataStub(self.metadata_database_channel)
        logger.info("Successfully connected to metadata database.")
        self.connected_to_metadata = True

    def register_training(self, training_set_size: int, num_workers: int) -> int:
        """
        Creates a new training object in the database with the given training_set_size and num_workers
        Returns:
            The id of the newly created training object
        """
        assert self.connected_to_metadata, "Tried to register training, but metadata server not connected."
        request = RegisterRequest(training_set_size=training_set_size, num_workers=num_workers)
        training_id = self.metadata_database.RegisterTraining(request).training_id
        return training_id

    def get_info_for_training(self, training_id: int) -> tuple[int, int]:
        """
        Queries the database for the the training set size and number of workers for a given training.

        Returns:
            Tuple of training set size and number of workers.
        """
        assert self.connected_to_metadata, "Tried to get training info, but metadata server not connected."

        request = GetTrainingRequest(training_id=training_id)
        info = self.metadata_database.GetTrainingInfo(request)
        training_set_size = info.training_set_size
        num_workers = info.num_workers
        return training_set_size, num_workers

    def get_samples_by_metadata_query(
        self, query: str
    ) -> tuple[list[str], list[float], list[bool], list[int], list[str]]:
        assert self.connected_to_metadata, "Tried to query metadata server, but metadata server not connected."
        request = GetByQueryRequest(query=query)
        samples = self.metadata_database.GetByQuery(request)
        return (samples.keys, samples.scores, samples.seen, samples.label, samples.data)
