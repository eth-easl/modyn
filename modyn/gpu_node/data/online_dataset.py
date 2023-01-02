from torch.utils.data import IterableDataset, get_worker_info
import typing
import torchvision
import json

from modyn.gpu_node.mocks.mock_selector_server import MockSelectorServer, GetSamplesRequest
from modyn.gpu_node.mocks.mock_storage_server import MockStorageServer, GetRequest


class OnlineDataset(IterableDataset):

    def __init__(self, training_id: int, transforms):
        self._training_id = training_id
        self._dataset_len = 0
        self._trainining_set_number = 0
        self._transforms = transforms
        self._deserialize_torchvision_transforms()

        # These mock the behavior of storage and selector servers.
        # TODO(fotstrt): remove them when the storage and selector grpc servers are fixed
        self._selectorstub = MockSelectorServer()
        self._storagestub = MockStorageServer()

    def _get_keys_from_selector(self, worker_id: int) -> list[str]:
        # TODO: replace this with grpc calls to the selector
        req = GetSamplesRequest(self._training_id, worker_id)
        samples_response = self._selectorstub.get_sample_keys(req)
        return samples_response.training_samples_subset

    def _get_data_from_storage(self, keys: list[str]) -> list[str]:
        # TODO: replace this with grpc calls to the selector
        req = GetRequest(keys=keys)
        data = self._storagestub.Get(req).value
        return data

    def _deserialize_torchvision_transforms(self):
        self._transform_list = []
        for transform in self._transforms:
            function = getattr(torchvision.transforms, transform['function'])
            args = json.loads(transform['args'])
            self._transform_list.append(function(**args))
        self._transform = torchvision.transforms.Compose(self._transform_list)

    def __iter__(self) -> typing.Iterator:
        worker_info = get_worker_info()
        if worker_info is None:
            # this is the main process. Give it worker_id 0
            worker_id = 0
        else:
            worker_id = worker_info.id
        self._trainining_set_number += 1

        keys = self._get_keys_from_selector(worker_id)
        raw_data = self._get_data_from_storage(keys)

        self._dataset_len = len(raw_data)

        processed_data = self._process(raw_data)
        return iter(processed_data)

    def __len__(self) -> int:
        return self._dataset_len

    def _process(self, data: list) -> list:
        processed_data = [self._transform(sample) for sample in data]
        return processed_data
