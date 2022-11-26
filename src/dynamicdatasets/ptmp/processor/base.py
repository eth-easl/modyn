from abc import ABC,abstractmethod
from threading import Thread
from concurrent import futures

import grpc

from dynamicdatasets.odm.odm_pb2 import SetRequest
from dynamicdatasets.odm.odm_pb2_grpc import ODMStub


class PostTrainingMetadataProcessor(ABC):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.__config = config
        self.__thread_pool = futures.ThreadPoolExecutor(max_workers=10)

    def process_post_training_metadata(self, training_id: str, data: str) -> None:
        self.__thread_pool.submit(self.__process_and_send, training_id, data)

    def __process_and_send(self, training_id: str, data: str) -> None:
        set_request = self._process_post_training_metadata(training_id, data)
        self.__send_to_odm(training_id, set_request)

    def __send_to_odm(self, training_id: str, set_request: SetRequest) -> None:
        channel = grpc.insecure_channel(self.__config['odm']['hostname'] +
            ':' +
            self.__config['odm']['port'])
        stub = ODMStub(channel)
        stub.Set(set_request)

    @abstractmethod
    def _process_post_training_metadata(self, training_id: str, data: str) -> SetRequest:
        """Processes post training metadata for the given training_id and data.

        Args:
            training_id: The training id.
            data: The data to process.

        Returns:
            The processed data.
        """
        raise NotImplementedError()
