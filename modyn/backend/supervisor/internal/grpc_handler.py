from modyn.storage.storage_pb2_grpc import StorageStub
from modyn.storage.storage_pb2 import DatasetAvailableRequest

import grpc
import logging

TIMEOUT_SEC = 5
logger = logging.getLogger(__name__)


class GRPCHandler():
    def __init__(self, modyn_config: dict):
        self.config = modyn_config
        self.connected_to_storage = False

        self.init_storage()

    def connection_established(self, channel) -> bool:
        try:
            grpc.channel_ready_future(channel).result(timeout=TIMEOUT_SEC)
            return True
        except grpc.FutureTimeoutError:
            return False

    def init_storage(self) -> None:
        assert self.config is not None
        storage_address = f"{self.config['storage']['hostname']}:{self.config['storage']['port']}"
        self.storage_channel = grpc.insecure_channel(storage_address)

        if not self.connection_established(self.storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at {storage_address}.")

        self.storage = StorageStub(self.storage_channel)
        logger.info("Successfully connected to storage.")
        self.connected_to_storage = True

    def dataset_available(self, dataset_id: str) -> bool:
        assert self.connected_to_storage, "Tried to check for dataset availability, but no storage connection."
        response = self.storage.CheckAvailability(DatasetAvailableRequest(dataset_id=dataset_id))

        return response.available
