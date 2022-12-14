from backend.selector.selector_pb2 import GetSamplesRequest
from backend.selector.selector_pb2_grpc import SelectorStub

from storage.storage_pb2 import GetRequest
from storage.storage_pb2_grpc import StorageStub

from torch.utils.data import IterableDataset, get_worker_info
import grpc
from abc import abstractmethod
import Typing

class OnlineDataset(IterableDataset):

    def __init__(self, training_id: int, config: dict):
        self._training_id = training_id
        self._config = config
        self._trainining_set_number = 0

    def __selector_stub(self) -> SelectorStub:
        selector_channel = grpc.insecure_channel(
            self._config['selector']['hostname'] +
            ':' +
            self._config['selector']['port'])
        return SelectorStub(selector_channel)

    def __storage_stub(self) -> StorageStub:
        storage_channel = grpc.insecure_channel(
            self._config['storage']['hostname'] +
            ':' +
            self._config['storage']['port'])
        return StorageStub(storage_channel)

    def _get_keys_from_selector(self, worker_id: int) -> list[str]:
        req = GetSamplesRequest(
            training_id=self._training_id,
            training_set_number=self._trainining_set_number,
            worker_id=worker_id)
        samples_response = self.__selector_stub().get_sample_keys(req)
        keys = samples_response.training_samples_subset
        return keys

    def _get_data_from_storage(self, keys: list[str]) -> list[str]:
        req = GetRequest(keys=keys)
        data = self.__storage_stub().Get(req).value
        return data

    def __iter__(self) -> Typing.iterator:
        worker_info = get_worker_info()
        if worker_info is None:
            # this is the main process. Give it worker_id 0
            worker_id = 0
        else:
            worker_id = worker_info.id
        self._trainining_set_number += 1
        keys = self._get_keys_from_selector(worker_id)
        raw_data = self._get_data_from_storage(keys)
        processed_data = self._process(raw_data)
        return iter(processed_data)

    def __len__(self) -> int:
        return self._config['trainer']['train_set_size']

    @abstractmethod
    def _process(self, data: list) -> list:
        """
        Override to add custom data processing.

        Args:
            data: sequence of elements from storage, most likely as json strings

        Returns:
            sequence of processed elements
        """
        raise NotImplementedError
