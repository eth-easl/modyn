import logging
import math
from collections.abc import Callable, Generator, Iterable

import grpc
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

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

logger = logging.getLogger(__name__)


# TODO(#275): inherit common abstraction of dataset
class FixedKeysDataset(IterableDataset):
    """The FixedKeysDataset is created given a list of fixed sample keys.

    It fetches samples by the given sample keys. It can be used when
    sample keys are known, but the corresponding trigger_id is unknown.
    It can also be used when user wants a dataset containing samples
    from multiple triggers if the keys are known. In DataDriftTrigger,
    for example, FixedKeysDataset is used for current untriggered
    samples because they belong to a future trigger whose trigger_id is
    unknown.
    """

    # pylint: disable=too-many-instance-attributes, abstract-method

    def __init__(
        self,
        dataset_id: str,
        bytes_parser: str,
        serialized_transforms: list[str],
        storage_address: str,
        keys: list[int],
        bytes_parser_target: str | None = None,
        serialized_target_transforms: list[str] | None = None,
        include_labels: bool = True,
        tokenizer: str | None = None,
        max_token_length: int =128

    ):
        self._dataset_id = dataset_id
        self._first_call = True
        self._include_labels = include_labels
        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._transform_list: list[Callable] = []
        self._transform: Callable | None = None
        self._storagestub: StorageStub = None
        self._bytes_parser_function: Callable | None = None
        self._serialized_target_transforms = serialized_target_transforms
        # Use target bytes parser if provided; otherwise, use the normal one.
        self._bytes_parser_target = bytes_parser_target if bytes_parser_target is not None else bytes_parser
        self._bytes_parser_function_target: Callable | None = None

        self._tokenizer = None
        self._tokenizer_name = tokenizer

        if tokenizer is not None:
            self._tokenizer = instantiate_class("modyn.models.tokenizers", tokenizer, max_token_length=max_token_length)

        self._keys = keys

        logger.debug("Initialized FixedKeysDataset.")

    def _init_transforms(self) -> None:
        self._bytes_parser_function = deserialize_function(self._bytes_parser, BYTES_PARSER_FUNC_NAME)
        self._bytes_parser_function_target = deserialize_function(self._bytes_parser_target, BYTES_PARSER_FUNC_NAME)
        self._transform = self._bytes_parser_function
        # Ensure _transform_target is always callable.
        self._transform_target = self._bytes_parser_function_target
        self._setup_composed_transform()
        self._setup_composed_target_transform()

    def _setup_composed_transform(self) -> None:
        assert self._bytes_parser_function is not None
        self._transform_list = [self._bytes_parser_function]
        for transform in self._serialized_transforms:
            function = eval(transform)  # pylint: disable=eval-used
            self._transform_list.append(function)
        if self._tokenizer is not None:
            self._transform_list.append(self._tokenizer)
        if self._transform_list:
            self._transform = transforms.Compose(self._transform_list)

    def _setup_composed_target_transform(self) -> None:
        target_transform_list = []
        if self._bytes_parser_function_target is not None:
            target_transform_list.append(self._bytes_parser_function_target)
        for transform in self._serialized_target_transforms or []:
            function = eval(transform)  # pylint: disable=eval-used
            target_transform_list.append(function)
        if self._tokenizer is not None:
            target_transform_list.append(self._tokenizer)
        if target_transform_list:
            self._transform_target = transforms.Compose(target_transform_list)

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

    def _info(self, msg: str, worker_id: int | None) -> None:  # pragma: no cover
        logger.info(f"[Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: int | None) -> None:  # pragma: no cover
        logger.debug(f"[Worker {worker_id}] {msg}")

    @staticmethod
    def _silence_pil() -> None:  # pragma: no cover
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)  # by default, PIL on DEBUG spams the console

    def _get_data_from_storage(self, keys: list[int]) -> Iterable[list[tuple[int, bytes, int]]]:
        request = GetRequest(dataset_id=self._dataset_id, keys=keys)
        response: GetResponse
        for response in self._storagestub.Get(request):
            if not self._include_labels:
                yield list(zip(response.keys, response.samples, response.target))
            else:
                yield list(zip(response.keys, response.samples, response.labels))

    def __iter__(self) -> Generator:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multi-threaded data loading. We use worker_id 0.
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

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
        keys_per_worker = int(math.ceil(total_samples / num_workers))
        worker_keys = self._keys[worker_id * keys_per_worker : min(total_samples, (worker_id + 1) * keys_per_worker)]

        # TODO(#175): we might want to do/accelerate prefetching here.
        for data in self._get_data_from_storage(worker_keys):
            for key, sample, label in data:
                if self._include_labels:
                    yield key, self._transform(sample), label
                else:
                    yield key, self._transform(sample), self._transform_target(label)
