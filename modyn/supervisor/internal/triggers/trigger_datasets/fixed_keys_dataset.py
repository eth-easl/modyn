import logging
import math
from typing import Callable, Generator, Iterable, Optional

import grpc
from modyn.storage.internal.grpc.generated.storage_pb2 import (  # pylint: disable=no-name-in-module
    GetRequest,
    GetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils import (
    BYTES_PARSER_FUNC_NAME,
    MAX_MESSAGE_SIZE,
    deserialize_function,
    grpc_connection_established,
    instantiate_class,
)
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

logger = logging.getLogger(__name__)


# TODO(#275): inherit common abstraction of dataset
class FixedKeysDataset(IterableDataset):
    # pylint: disable=too-many-instance-attributes, abstract-method
    def __init__(
        self,
        dataset_id: str,
        bytes_parser: str,
        serialized_transforms: list[str],
        storage_address: str,
        trigger_id: int,
        keys: list[int],
        tokenizer: Optional[str] = None,
    ):
        self._trigger_id = trigger_id
        self._dataset_id = dataset_id
        self._first_call = True

        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._transform_list: list[Callable] = []
        self._transform: Optional[Callable] = None
        self._storagestub: StorageStub = None
        self._bytes_parser_function: Optional[Callable] = None

        # tokenizer for NLP tasks
        self._tokenizer = None
        self._tokenizer_name = tokenizer
        if tokenizer is not None:
            self._tokenizer = instantiate_class("modyn.models.tokenizers", tokenizer)

        self._keys = keys

        logger.debug("Initialized FixedKeysDataset.")

    def _init_transforms(self) -> None:
        self._bytes_parser_function = deserialize_function(self._bytes_parser, BYTES_PARSER_FUNC_NAME)
        self._transform = self._bytes_parser_function
        self._setup_composed_transform()

    def _setup_composed_transform(self) -> None:
        assert self._bytes_parser_function is not None

        self._transform_list = [self._bytes_parser_function]
        for transform in self._serialized_transforms:
            function = eval(transform)  # pylint: disable=eval-used
            self._transform_list.append(function)

        if self._tokenizer is not None:
            self._transform_list.append(self._tokenizer)

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
        logger.info(f"[Trigger {self._trigger_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.debug(f"[Trigger {self._trigger_id}][Worker {worker_id}] {msg}")

    @staticmethod
    def _silence_pil() -> None:  # pragma: no cover
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)  # by default, PIL on DEBUG spams the console

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
        else:
            worker_id = worker_info.id

        if self._first_call:
            self._first_call = False
            self._debug("This is the first run of iter, making gRPC connections.", worker_id)
            # We have to initialize transformations and gRPC connections here to do it per dataloader worker,
            # otherwise the transformations/gRPC connections cannot be pickled for the new processes.
            self._init_transforms()
            self._init_grpc()
            FixedKeysDataset._silence_pil()
            self._debug("gRPC initialized.", worker_id)

        assert self._transform is not None

        total_samples = len(self._keys)
        keys_per_worker = int(math.ceil(total_samples / worker_info.num_workers))
        worker_keys = self._keys[worker_id * keys_per_worker : min(total_samples, (worker_id + 1) * keys_per_worker)]

        # TODO(#175): we might want to do/accelerate prefetching here.
        for data in self._get_data_from_storage(worker_keys):
            for key, sample, label in data:
                yield key, self._transform(sample), label
