import typing

from modyn.trainer_server.internal.mocks.mock_selector_server import GetSamplesRequest, MockSelectorServer
from modyn.trainer_server.internal.mocks.mock_storage_server import GetRequest, MockStorageServer
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms


class OnlineDataset(IterableDataset):
    # pylint: disable=too-many-instance-attributes, abstract-method

    def __init__(
        self,
        training_id: int,
        dataset_id: str,
        serialized_transforms: list[str],
        train_until_sample_id: str,
    ):
        self._training_id = training_id
        self._dataset_id = dataset_id
        self._dataset_len = 0
        self._trainining_set_number = 0
        self._serialized_transforms = serialized_transforms
        self._transform = lambda x: x  # identity as default
        self._deserialize_torchvision_transforms()
        self._train_until_sample_id = train_until_sample_id

        # These mock the behavior of storage and selector servers.
        # TODO(#74): remove them when the storage and selector grpc servers are fixed
        self._selectorstub = MockSelectorServer()
        self._storagestub = MockStorageServer()

    def _get_keys_from_selector(self, worker_id: int) -> list[str]:
        # TODO(#74): replace this with grpc calls to the selector
        req = GetSamplesRequest(self._training_id, self._train_until_sample_id, worker_id)
        samples_response = self._selectorstub.get_sample_keys(req)
        return samples_response.training_samples_subset

    def _get_data_from_storage(self, keys: list[str]) -> tuple[list[str], list[typing.Any]]:
        # TODO(#74): replace this with grpc calls to the selector
        req = GetRequest(dataset_id=self._dataset_id, keys=keys)
        response = self._storagestub.Get(req)
        return response.samples, response.labels

    def _deserialize_torchvision_transforms(self) -> None:
        self._transform_list = []
        for transform in self._serialized_transforms:
            function = eval(transform)  # pylint: disable=eval-used
            self._transform_list.append(function)
        if len(self._transform_list) > 0:
            self._transform = transforms.Compose(self._transform_list)

    def __iter__(self) -> typing.Generator:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multithreaded data loading. We use worker_id 0.
            worker_id = 0
        else:
            worker_id = worker_info.id
        self._trainining_set_number += 1

        keys = self._get_keys_from_selector(worker_id)
        data, labels = self._get_data_from_storage(keys)

        self._dataset_len = len(data)

        for sample, label in zip(data, labels):
            yield self._transform(sample), label

    def __len__(self) -> int:
        return self._dataset_len
