import gc
import logging
from typing import Callable, Generator, Optional

import grpc

# pylint: disable-next=no-name-in-module
from modyn.selector.internal.grpc.generated.selector_pb2 import (
    GetNumberOfPartitionsRequest,
    GetSamplesRequest,
    NumberOfPartitionsResponse,
)
from modyn.selector.internal.grpc.generated.selector_pb2_grpc import SelectorStub
from modyn.storage.internal.grpc.generated.storage_pb2 import (  # pylint: disable=no-name-in-module
    GetRequest,
    GetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.utils.utils import (
    BYTES_PARSER_FUNC_NAME,
    MAX_MESSAGE_SIZE,
    deserialize_function,
    flatten,
    grpc_connection_established,
)
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
        self._first_call = True

        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._selector_address = selector_address
        self._transform_list: list[Callable] = []
        self._transform: Optional[Callable] = None
        self._storagestub: StorageStub = None
        self._selectorstub: SelectorStub = None
        self._bytes_parser_function: Optional[Callable] = None
        self._num_partitions = 0

        logger.debug("Initialized OnlineDataset.")

    # pylint: disable=unused-argument
    def _get_keys_from_selector(self, worker_id: int, partition_id: int) -> list[int]:
        assert self._selectorstub is not None

        req = GetSamplesRequest(
            pipeline_id=self._pipeline_id, trigger_id=self._trigger_id, worker_id=worker_id, partition_id=partition_id
        )
        # TODO(#138): take into account sample weights when needed
        return flatten(
            [response.training_samples_subset for response in self._selectorstub.get_sample_keys_and_weights(req)]
        )

    def _get_data_from_storage(self, selector_keys: list[int]) -> tuple[list[bytes], list[int]]:
        req = GetRequest(dataset_id=self._dataset_id, keys=selector_keys)

        data_from_storage: dict[int, tuple[bytes, int]] = {}
        response: GetResponse
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
        self._bytes_parser_function = deserialize_function(self._bytes_parser, BYTES_PARSER_FUNC_NAME)
        self._transform = self._bytes_parser_function
        self._deserialize_torchvision_transforms()

    def _init_grpc(self) -> None:
        selector_channel = grpc.insecure_channel(
            self._selector_address,
            options=[
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        if not grpc_connection_established(selector_channel):
            raise ConnectionError(
                f"Could not establish gRPC connection to selector at address {self._selector_address}."
            )
        self._selectorstub = SelectorStub(selector_channel)

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

    def _silence_pil(self) -> None:  # pragma: no cover
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)  # by default, PIL on DEBUG spams the console

    def _info(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.info(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.debug(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _get_data(self, worker_id: int, partition_id: int) -> tuple[list[int], list[bytes], list[int]]:
        self._info("Getting keys from selector", worker_id)
        keys = self._get_keys_from_selector(worker_id, partition_id)
        self._info("Getting data from storage", worker_id)
        data, labels = self._get_data_from_storage(keys)
        return keys, data, labels

    def _get_num_data_partitions(self) -> int:
        assert self._selectorstub is not None

        num_partitions_request = GetNumberOfPartitionsRequest(
            pipeline_id=self._pipeline_id,
            trigger_id=self._trigger_id,
        )

        response: NumberOfPartitionsResponse = self._selectorstub.get_number_of_partitions(num_partitions_request)
        return response.num_partitions

    # pylint: disable=too-many-locals
    def __iter__(self) -> Generator:
        worker_info = get_worker_info()
        if worker_info is None:
            # Non-multithreaded data loading. We use worker_id 0.
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
            self._silence_pil()
            self._debug("gRPC initialized.", worker_id)

        assert self._transform is not None
        self._num_partitions = self._get_num_data_partitions()
        self._info(f"Total number of partitions will be {self._num_partitions}", worker_id)

        keys, data, labels = self._get_data(worker_id=worker_id, partition_id=0)

        for partition in range(self._num_partitions):
            num_samples_on_this_partition = len(keys)
            # We (arbitrarily) fetch the next partition when we have seen 80% of the current partition
            fetch_next_partition_idx = int(num_samples_on_this_partition * 0.8)
            self._info(f"Train on partition {partition}, on {num_samples_on_this_partition} batches", worker_id)

            for idx, (key, sample, label) in enumerate(zip(keys, data, labels)):
                if partition < self._num_partitions - 1 and idx == fetch_next_partition_idx:
                    # TODO(#175) in case this blocks training
                    new_keys, new_data, new_labels = self._get_data(worker_id=worker_id, partition_id=partition + 1)
                # mypy complains here because _transform has unknown type, which is ok
                yield key, self._transform(sample), label  # type: ignore

            # this should mean we keep only two partitions in mem
            if partition < self._num_partitions - 1:
                del keys
                del data
                del labels
                keys, data, labels = new_keys, new_data, new_labels
                del new_keys
                del new_data
                del new_labels
                gc.collect()
