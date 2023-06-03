import logging
from inspect import isfunction
from typing import Any, Callable, Generator, Iterable, Optional

import grpc
from modyn.storage.internal.grpc.generated.storage_pb2 import (  # pylint: disable=no-name-in-module
    GetDataPerWorkerRequest,
    GetDataPerWorkerResponse,
    GetRequest,
    GetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils.utils import MAX_MESSAGE_SIZE, grpc_connection_established
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

logger = logging.getLogger(__name__)


class EvaluationDataset(IterableDataset):
    # pylint: disable=too-many-instance-attributes, abstract-method

    def __init__(
        self,
        dataset_id: str,
        bytes_parser: str,
        serialized_transforms: list[str],
        storage_address: str,
        evaluation_id: int,
    ):
        self._evaluation_id = evaluation_id
        self._dataset_id = dataset_id
        self._first_call = True
        self._mod_dict: dict[str, Any] = {}

        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._transform_list: list[Callable] = []
        self._transform: Optional[Callable] = None
        self._storagestub: StorageStub = None
        self._bytes_parser_function: Optional[Callable] = None

        logger.debug("Initialized EvaluationDataset.")

    def _init_transforms(self) -> None:
        exec(self._bytes_parser, self._mod_dict)  # pylint: disable=exec-used
        if "bytes_parser_function" not in self._mod_dict or not isfunction(self._mod_dict["bytes_parser_function"]):
            raise ValueError("Missing function bytes_parser_function from evaluation invocation")
        self._bytes_parser_function = self._mod_dict["bytes_parser_function"]
        self._transform = self._bytes_parser_function
        self._deserialize_torchvision_transforms()

    def _deserialize_torchvision_transforms(self) -> None:
        assert self._bytes_parser_function is not None

        self._transform_list = [self._bytes_parser_function]
        for transform in self._serialized_transforms:
            function = eval(transform)  # pylint: disable=eval-used
            self._transform_list.append(function)
        if len(self._transform_list) > 0:
            self._transform = transforms.Compose(self._transform_list)

    def _init_grpc(self) -> None:
        storage_channel = grpc.insecure_channel(
            self._storage_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        if not grpc_connection_established(storage_channel):
            raise ConnectionError(f"Could not establish gRPC connection to storage at address {self._storage_address}.")
        self._storagestub = StorageStub(storage_channel)

    def _info(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.info(f"[Evaluation {self._evaluation_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.debug(f"[Evaluation {self._evaluation_id}][Worker {worker_id}] {msg}")

    @staticmethod
    def _silence_pil() -> None:  # pragma: no cover
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)  # by default, PIL on DEBUG spams the console

    def _get_keys_from_storage(self, worker_id: int, total_workers: int) -> Iterable[list[int]]:
        self._info("Getting keys from storage", worker_id)
        req_keys = GetDataPerWorkerRequest(
            dataset_id=self._dataset_id, worker_id=worker_id, total_workers=total_workers
        )
        resp_keys: GetDataPerWorkerResponse
        for resp_keys in self._storagestub.GetDataPerWorker(req_keys):
            yield resp_keys.keys

    def _get_data_from_storage(self, keys: list[int]) -> Iterable[list[tuple[int, bytes, int]]]:
        request = GetRequest(dataset_id=self._dataset_id, keys=keys)
        response: GetResponse
        for response in self._storagestub.Get(request):
            yield list(zip(response.keys, response.samples, response.labels))

    def __iter__(self) -> Generator:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multi-threaded data loading. We use worker_id 0.
            worker_id = 0
            total_workers = 1
        else:
            worker_id = worker_info.id
            total_workers = worker_info.num_workers

        if self._first_call:
            self._first_call = False
            self._debug("This is the first run of iter, making gRPC connections.", worker_id)
            # We have to initialize transformations and gRPC connections here to do it per dataloader worker,
            # otherwise the transformations/gRPC connections cannot be pickled for the new processes.
            self._init_transforms()
            self._init_grpc()
            EvaluationDataset._silence_pil()
            self._debug("gRPC initialized.", worker_id)

        assert self._transform is not None

        for keys in self._get_keys_from_storage(worker_id, total_workers):
            for data in self._get_data_from_storage(keys):
                for key, sample, label in data:
                    yield key, self._transform(sample), label
