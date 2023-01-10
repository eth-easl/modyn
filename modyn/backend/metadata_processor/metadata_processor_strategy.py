from abc import ABC, abstractmethod

from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataStub
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import SetRequest


class MetadataProcessorStrategy(ABC):
	"""This class is the base class for Metadata Processors. In order to extend
	this class to perform custom processing, implement _process_post_training_metadata
	"""
	
	def __init__(self, config: dict):
		self.config = config

	def process_post_training_metadata(
			self, training_id: int, serialized_data: str) -> None:
		"""
		Process the metadata and save it to the Metadata Database.

		Args:
			training_id (int): The training ID.
			data (str): Serialized post training metadata.
		"""
		data = self._process_post_training_metadata(training_id, serialized_data)
		self.write_to_database(training_id, data)

	def write_to_database(self, training_id: int, data: dict) -> None:
		"""
        Write data to metadata database.

        Args:
            training_id (str): The training id.
            set_request (dict): The metadata.
        """
        set_request = SetRequest(training_id=training_id, data=data['data'],
        	keys=data['keys'], scores=data['scores'])
		channel = grpc.insecure_channel(self.__config['metadata_database']['hostname'] +
                                        ':' +
                                        self.__config['metadata_database']['port'])

        stub = MetadataStub(channel)
        stub.Set(set_request)

	@abstractmethod
	def _process_post_training_metadata(self, training_id: int, data: str) -> dict:
		raise NotImplementedError()