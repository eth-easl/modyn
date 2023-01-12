import logging

import grpc

# pylint: disable-next=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2 import (
    DatasetAvailableRequest,
    GetDataInIntervalRequest,
    GetDataInIntervalResponse,
    GetNewDataSinceRequest,
    GetNewDataSinceResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils import current_time_millis, grpc_connection_established

logger = logging.getLogger(__name__)


class GRPCHandler:
    def __init__(self, modyn_config: dict):
        self.config = modyn_config
        self.connected_to_storage = False

        self.init_storage()

    def init_storage(self) -> None:
        assert self.config is not None
        storage_address = f"{self.config['storage']['hostname']}:{self.config['storage']['port']}"
        self.storage_channel = grpc.insecure_channel(storage_address)

        if not grpc_connection_established(self.storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at {storage_address}.")

        self.storage = StorageStub(self.storage_channel)
        logger.info("Successfully connected to storage.")
        self.connected_to_storage = True

    def dataset_available(self, dataset_id: str) -> bool:
        assert self.connected_to_storage, "Tried to check for dataset availability, but no storage connection."
        response = self.storage.CheckAvailability(DatasetAvailableRequest(dataset_id=dataset_id))

        return response.available

    def get_new_data_since(self, dataset_id: str, timestamp: int) -> list[tuple[str, int]]:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        request = GetNewDataSinceRequest(dataset_id=dataset_id, timestamp=timestamp)
        response: GetNewDataSinceResponse = self.storage.GetNewDataSince(request)

        keys = response.keys
        # TODO(#76): We want to return the keys _and_ the timestamp of the data from the storage.
        # For now, manual timestamp 42
        keys_timestamped = [(key, 42) for key in keys]
        return keys_timestamped

    def get_data_in_interval(self, dataset_id: str, start_timestamp: int, end_timestamp: int) -> list[tuple[str, int]]:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        request = GetDataInIntervalRequest(
            dataset_id=dataset_id,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        response: GetDataInIntervalResponse = self.storage.GetDataInInterval(request)

        keys = response.keys

        # TODO(#76): We want to return the keys _and_ the timestamp of the data from the storage.
        #  For now, manual timestamp 42
        keys_timestamped = [(key, 42) for key in keys]
        return keys_timestamped

    def get_time_at_storage(self) -> int:
        if not self.connected_to_storage:
            raise ConnectionError("Tried to fetch data from storage, but no connection was made.")

        # TODO(#77): Implement gRPC call.
        return current_time_millis()

    def register_pipeline_at_selector(self, pipeline_config: dict) -> int:
        # TODO(#64): Implement gRPC call.
        return 42

    def unregister_pipeline_at_selector(self, pipeline_id: int) -> None:
        # TODO(#64): Implement gRPC call.
        pass

    def inform_selector(self, pipeline_id: int, data: list[tuple[str, int]]) -> None:
        # TODO(#64): Implement gRPC call.
        pass

    def inform_selector_and_trigger(self, pipeline_id: int, data: list[tuple[str, int]]) -> None:
        # TODO(#64): Implement gRPC call.
        pass

    def trainer_server_available(self) -> bool:
        # TODO(#78): implement.
        return True

    def shutdown_trainer_server(self, training_id: int) -> None:
        # TODO(#78): implement.
        pass

    def start_trainer_server(self, pipeline_id: int, trigger_id: int, pipeline_config: dict) -> int:
        # TODO(#78): implement.
        return 42

    def wait_for_training_completion(self, training_id: int) -> None:
        # TODO(#78): implement.
        pass
