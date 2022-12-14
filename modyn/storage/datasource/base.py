from abc import abstractmethod, ABC
import time

import grpc

from storage.storage_pb2_grpc import StorageStub
from storage.storage_pb2 import PutRequest


class BaseSource(ABC):

    def __init__(self, config: dict):
        self._config = config

    @abstractmethod
    def get_next(self, limit: int) -> tuple[list[str], list[str]]:
        """
        Get next data from the source

        Returns:
            tuple[list[bytes], list[bytes]]: tuple of keys and data
        """
        raise NotImplementedError

    def add_to_storage(self, keys: list[str], data: list[str]):
        """
        Add data to the storage
        """
        storage_channel = grpc.insecure_channel(
            self._config['storage']['hostname'] +
            ':' +
            self._config['storage']['port'])
        storage_stub = StorageStub(storage_channel)
        storage_stub.Put(PutRequest(keys=keys, value=data))

    def run(self):
        """
        Run the source
        """
        while True:
            keys, data = self.get_next(
                self._config['storage']['data_source']['batch_size'])
            self.add_to_storage(keys, data)
            time.sleep(self._config['storage']
                       ['data_source']['batch_interval'])
