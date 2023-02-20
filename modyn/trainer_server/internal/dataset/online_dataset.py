import logging
from inspect import isfunction
from typing import Any, Callable, Generator, Optional

import grpc

# pylint: disable-next=no-name-in-module
from modyn.backend.selector.internal.grpc.generated.selector_pb2 import GetSamplesRequest, SamplesResponse
from modyn.backend.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2 import GetRequest  # pylint: disable=no-name-in-module
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils.utils import grpc_connection_established
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

logger = logging.getLogger(__name__)


class OnlineDataset(IterableDataset):
    # pylint: disable=too-many-instance-attributes, abstract-method

    def __init__(
        self,
        pipeline_id: int,
        trigger_id: int,
        dataset_id: str,
        bytes_parser: str,
        serialized_transforms: list[str],
        storage_address: str,
        selector_address: str,
        training_id: int,
    ):
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id
        self._training_id = training_id
        self._dataset_id = dataset_id
        self._dataset_len = 0
        self._trainining_set_number = 0
        self._mod_dict: dict[str, Any] = {}

        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._selector_address = selector_address
        self._transform_list: list[Callable] = []
        self._transform: Optional[Callable] = None
        self._storagestub: StorageStub = None
        self._selectorstub: SelectorStub = None
        self._bytes_parser_function: Optional[Callable] = None

        logger.debug("Initialized OnlineDataset.")

    def _get_keys_from_selector(self, worker_id: int) -> list[str]:
        assert self._selectorstub is not None

        req = GetSamplesRequest(pipeline_id=self._pipeline_id, trigger_id=self._trigger_id, worker_id=worker_id)
        samples_response: SamplesResponse = self._selectorstub.get_sample_keys_and_weights(req)
        return samples_response.training_samples_subset  # TODO(#138): take into account sample weights when needed

    def _get_data_from_storage(self, selector_keys: list[str]) -> tuple[list[bytes], list[int]]:
        req = GetRequest(dataset_id=self._dataset_id, keys=selector_keys)

        data_from_storage: dict[str, tuple[bytes, int]] = {}
        for _, response in enumerate(self._storagestub.Get(req)):
            for key, sample, label in zip(response.keys, response.samples, response.labels):
                data_from_storage[key] = (sample, label)

        sample_list = [data_from_storage[key][0] for key in selector_keys]
        label_list = [data_from_storage[key][1] for key in selector_keys]

        return sample_list, label_list

    def _deserialize_torchvision_transforms(self) -> None:
        assert self._bytes_parser_function is not None
        
        self._transform_list = [self._bytes_parser_function]
        for transform in self._serialized_transforms:
            function = eval(transform)  # pylint: disable=eval-used
            self._transform_list.append(function)
        if len(self._transform_list) > 0:
            self._transform = transforms.Compose(self._transform_list)

    def _init_transforms(self) -> None:
        exec(self._bytes_parser, self._mod_dict)  # pylint: disable=exec-used
        if "bytes_parser_function" not in self._mod_dict or not isfunction(self._mod_dict["bytes_parser_function"]):
            raise ValueError("Missing function bytes_parser_function from training invocation")
        self._bytes_parser_function = self._mod_dict["bytes_parser_function"]
        self._transform = self._bytes_parser_function
        self._deserialize_torchvision_transforms()

    def _init_grpc(self) -> None:
        selector_channel = grpc.insecure_channel(self._selector_address)
        if not grpc_connection_established(selector_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to selector at address {self._selector_address}."
            )
        self._selectorstub = SelectorStub(selector_channel)

        storage_channel = grpc.insecure_channel(self._storage_address)
        if not grpc_connection_established(storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at address {self._storage_address}.")
        self._storagestub = StorageStub(storage_channel)

    def _silence_pil(self) -> None:
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)  # by default, PIL on DEBUG spams the console

    def _info(self, msg: str, worker_id: Optional[int]) -> None:
        logger.info(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: Optional[int]) -> None:
        logger.debug(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def __iter__(self) -> Generator:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multithreaded data loading. We use worker_id 0.
            worker_id = 0
        else:
            worker_id = worker_info.id

        if self._trainining_set_number == 0:
            self._debug("This is the first run of iter, making gRPC connections.", worker_id)
            # We have to initialize transformations and gRPC connections here to do it per dataloader worker,
            # otherwise the transformations/gRPC connections cannot be pickled for the new processes.
            self._init_transforms()
            self._init_grpc()
            self._silence_pil()
            self._debug("gRPC initialized.", worker_id)

        assert self._transform is not None
        self._trainining_set_number += 1

        self._info("Getting keys from selector", worker_id)
        keys = self._get_keys_from_selector(worker_id)
        self._info("Getting data from storage", worker_id)
        # TODO(#149): Optimize this
        data, labels = self._get_data_from_storage(keys)

        self._dataset_len = len(data)
        self._info("Data obtained (len = {self._dataset_len})", worker_id)

        for key, sample, label in zip(keys, data, labels):
            # mypy complains here because _transform has unknown type, which is ok
            yield key, self._transform(sample), label  # type: ignore

    def __len__(self) -> int:
        return self._dataset_len
