from inspect import isfunction
from typing import Any, Generator

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
        bytes_parser: str,
        serialized_transforms: list[str],
        train_until_sample_id: str,
    ):
        self._training_id = training_id
        self._dataset_id = dataset_id
        self._dataset_len = 0
        self._trainining_set_number = 0
        mod_dict: dict[str, Any] = {}
        exec(bytes_parser, mod_dict)  # pylint: disable=exec-used
        if "bytes_parser_function" not in mod_dict or not isfunction(mod_dict["bytes_parser_function"]):
            raise ValueError("Missing function bytes_parser_function from training invocation")
        self._bytes_parser_function = mod_dict["bytes_parser_function"]
        self._serialized_transforms = serialized_transforms
        self._transform = self._bytes_parser_function
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

    def _get_data_from_storage(self, keys: list[str]) -> tuple[list[str], list[Any]]:
        # TODO(#74): replace this with grpc calls to the selector
        req = GetRequest(dataset_id=self._dataset_id, keys=keys)
        response = self._storagestub.Get(req)
        return response.samples, response.labels

    def _deserialize_torchvision_transforms(self) -> None:
        self._transform_list = [self._bytes_parser_function]
        for transform in self._serialized_transforms:
            function = eval(transform)  # pylint: disable=eval-used
            self._transform_list.append(function)
        if len(self._transform_list) > 0:
            self._transform = transforms.Compose(self._transform_list)

    def __iter__(self) -> Generator:
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
