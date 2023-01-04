from abc import ABC, abstractmethod
from concurrent import futures

from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2_grpc import MetadataStub
from modyn.backend.metadata_database.internal.grpc.generated.metadata_pb2 import SetRequest


class PostTrainingMetadataProcessor(ABC):
    """
    This method is called when the PostTrainingMetadataProcessor receives a
    PostTrainingMetadataRequest. It should process the data and return a
    SetRequest.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.__config = config
        self.__thread_pool = futures.ThreadPoolExecutor(max_workers=10)
        self.grpc = GRPCHandler(config)

    def process_post_training_metadata(
            self, training_id: int, data: str) -> None:
        """
        Add the task to process the post training metadata and write it to database to the thread pool.

        Args:
            training_id (str): The training id.
            data (str): The post training metadata.
        """
        self.__thread_pool.submit(self.__process_and_send, training_id, data)

    def __process_and_save(self, training_id: int, data: str) -> None:
        """
        Process the post training metadata and write it to the metadata database.

        Args:
            training_id (str): The training id.
            data (str): The post training metadata.
        """
        enriched_data = self._process_post_training_metadata(training_id, data)
        self.__write_to_db(training_id, enriched_data)

    def __write_to_db(self, set_request: SetRequest) -> None:
        """
        Write data to metadata database.

        Args:
            training_id (str): The training id.
            set_request (dict): The metadata.
        """
        channel = grpc.insecure_channel(self.__config['metadata_database']['hostname'] +
                                        ':' +
                                        self.__config['metadata_database']['port'])

        stub = MetadataStub(channel)
        stub.Set(set_request)

    @abstractmethod
    def _process_post_training_metadata(
            self, training_id: int, data: str) -> SetRequest:
        """Processes post training metadata for the given training_id and data.

        Args:
            training_id: The training id.
            data: The data to process.

        Returns:
            The processed data
        """
        raise NotImplementedError()
