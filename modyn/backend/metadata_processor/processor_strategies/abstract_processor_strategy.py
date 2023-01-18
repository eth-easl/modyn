from abc import ABC, abstractmethod

import grpc
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import SetRequest
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataStub


class AbstractProcessorStrategy(ABC):
    """This class is the base class for Metadata Processors. In order to extend
    this class to perform custom processing, implement process_metadata
    """

    def __init__(self, modyn_config: dict):
        self.config = modyn_config

    def process_post_training_metadata(self, training_id: int, serialized_data: str) -> None:
        """
        Process the metadata and save it to the Metadata Database.

        Args:
            training_id (int): The training ID.
            data (str): Serialized post training metadata.
        """
        data = self.process_metadata(training_id, serialized_data)
        self.write_to_database(training_id, data)

    def write_to_database(self, training_id: int, data: dict) -> None:
        """
        Write data to metadata database.

        Args:
            training_id (str): The training id.
            set_request (dict): The metadata.
        """
        set_request = SetRequest(
            training_id=training_id,
            keys=data["keys"],
            seen=data["seen"],
            data=data["data"],
        )

        channel = grpc.insecure_channel(
            self.config["metadata_database"]["hostname"] + ":" + self.config["metadata_database"]["port"]
        )
        stub = MetadataStub(channel)
        stub.Set(set_request)

    @abstractmethod
    def process_metadata(self, training_id: int, data: str) -> dict:
        raise NotImplementedError()
