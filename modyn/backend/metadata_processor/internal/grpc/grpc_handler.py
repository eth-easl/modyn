import logging

import grpc

# pylint: disable-next=no-name-in-module
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import SetRequest
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataStub
from modyn.utils import grpc_connection_established

logger = logging.getLogger(__name__)


class GRPCHandler:
    def __init__(self, modyn_config: dict):
        self.config = modyn_config
        self.connected_to_database = False

        self.init_connection()

    def init_connection(self) -> None:
        assert self.config is not None
        database_address = f"{self.config['metadata_database']['hostname']}:{self.config['metadata_database']['port']}"
        self.database_channel = grpc.insecure_channel(database_address)

        if not grpc_connection_established(self.database_channel):
            raise ConnectionError(f"Could not establish gRPC connection to database at {database_address}.")

        self.database = MetadataStub(self.database_channel)
        logger.info("Successfully connected to metadata database.")
        self.connected_to_database = True

    def set_metadata(self, pipeline_id: int, trigger_id: int, data: str) -> None:
        assert self.connected_to_database, "Tried to write metadata, but no database connection."
        # TODO(): change this based on Metadata Database new interface
        # self.database.Set(SetRequest(training_id=training_id, keys=data["keys"], seen=data["seen"], data=data["data"]))
