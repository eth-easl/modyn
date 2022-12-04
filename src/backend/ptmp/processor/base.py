from abc import ABC, abstractmethod
from threading import Thread
from concurrent import futures

import grpc

from backend.odm.odm_pb2 import SetRequest
from backend.odm.odm_pb2_grpc import ODMStub


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

    def process_post_training_metadata(
            self, training_id: str, data: str) -> None:
        """
        Add the task to process the post training metadata and send it to the ODM to the thread pool.

        Args:
            training_id (str): The training id.
            data (str): The post training metadata.
        """
        self.__thread_pool.submit(self.__process_and_send, training_id, data)

    def __process_and_send(self, training_id: str, data: str) -> None:
        """
        Process the post training metadata and send it to the ODM.

        Args:
            training_id (str): The training id.
            data (str): The post training metadata.
        """
        set_request = self._process_post_training_metadata(training_id, data)
        self.__send_to_odm(training_id, set_request)

    def __send_to_odm(self, training_id: str, set_request: SetRequest) -> None:
        """
        Send the set request to the ODM.

        Args:
            training_id (str): The training id.
            set_request (SetRequest): The set request.
        """
        channel = grpc.insecure_channel(self.__config['odm']['hostname'] +
                                        ':' +
                                        self.__config['odm']['port'])
        stub = ODMStub(channel)
        stub.Set(set_request)

    @abstractmethod
    def _process_post_training_metadata(
            self, training_id: str, data: str) -> SetRequest:
        """Processes post training metadata for the given training_id and data.

        Args:
            training_id: The training id.
            data: The data to process.

        Returns:
            The processed data.
        """
        raise NotImplementedError()
