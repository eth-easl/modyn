import gc
import logging
from typing import Callable, Generator, Optional, Tuple, Union

import grpc
from modyn.storage.internal.grpc.generated.storage_pb2 import (  # pylint: disable=no-name-in-module
    GetRequest,
    GetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.dataset.key_sources import AbstractKeySource, SelectorKeySource
from modyn.utils.utils import (
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
        tokenizer: Optional[str],
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
        self._bytes_parser_function: Optional[Callable] = None
        self._num_partitions = 0

        # the default key source is the Selector. Then it can be changed using change_key_source
        self._key_source = SelectorKeySource(self._pipeline_id, self._trigger_id, self._selector_address)
        self._uses_weights = None

        # tokenizer for NLP tasks
        self._tokenizer = None
        self._tokenizer_name = tokenizer
        if tokenizer is not None:
            self._tokenizer = instantiate_class("modyn.models.tokenizers", tokenizer)

        logger.debug("Initialized OnlineDataset.")

    def change_key_source(self, source: AbstractKeySource) -> None:
        self._key_source = source

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

    def _init_transforms(self) -> None:
        self._bytes_parser_function = deserialize_function(self._bytes_parser, BYTES_PARSER_FUNC_NAME)
        self._transform = self._bytes_parser_function
        self._setup_composed_transform()

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

    def _silence_pil(self) -> None:  # pragma: no cover
        pil_logger = logging.getLogger("PIL")
        pil_logger.setLevel(logging.INFO)  # by default, PIL on DEBUG spams the console

    def _info(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.info(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: Optional[int]) -> None:  # pragma: no cover
        logger.debug(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _get_data(
        self, worker_id: int, partition_id: int
    ) -> tuple[list[int], list[bytes], list[int], Optional[list[float]]]:
        keys, weights = self._key_source.get_keys_and_weights(worker_id, partition_id)

        self._info("Getting data from storage", worker_id)
        data, labels = self._get_data_from_storage(keys)
        return keys, data, labels, weights

    def _get_data_iterator(
        self, keys: list[int], data: list[bytes], labels: list[int], weights: Optional[list[float]]
    ) -> enumerate:
        assert self._uses_weights is not None

        # pylint: disable-next = unsubscriptable-object
        iterator: Union[zip[Tuple[int, bytes, int]], zip[Tuple[int, bytes, int, float]]]
        if self._uses_weights:
            assert weights is not None and len(weights) == len(keys)
            iterator = zip(keys, data, labels, weights)
        else:
            iterator = zip(keys, data, labels)
        return enumerate(iterator)

    def _unpack_data_tuple(self, data_tuple: Tuple) -> Tuple[int, bytes, int, Optional[float]]:
        assert self._uses_weights is not None

        if self._uses_weights:
            key, sample, label, weight = data_tuple
        else:
            key, sample, label = data_tuple
            weight = None

        return key, sample, label, weight

    def _get_data_tuple(self, key: int, sample: bytes, label: int, weight: Optional[float]) -> Optional[Tuple]:
        assert self._uses_weights is not None
        # mypy complains here because _transform has unknown type, which is ok
        tranformed_sample = self._transform(sample)  # type: ignore

        if self._uses_weights:
            return key, tranformed_sample, label, weight
        return key, tranformed_sample, label

    def end_of_trigger_cleaning(self) -> None:
        self._key_source.end_of_trigger_cleaning()

    # pylint: disable=too-many-locals, too-many-branches

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
            self._key_source.init_worker()
            self._uses_weights = self._key_source.uses_weights()
            self._silence_pil()
            self._debug("gRPC initialized.", worker_id)

        assert self._transform is not None
        self._num_partitions = self._key_source.get_num_data_partitions()
        self._info(f"Total number of partitions will be {self._num_partitions}", worker_id)

        keys, data, labels, weights = self._get_data(worker_id=worker_id, partition_id=0)

        for partition in range(self._num_partitions):
            num_samples_on_this_partition = len(keys)
            # We (arbitrarily) fetch the next partition when we have seen 80% of the current partition
            fetch_next_partition_idx = int(num_samples_on_this_partition * 0.8)
            self._info(f"Train on partition {partition}, on {num_samples_on_this_partition} batches", worker_id)

            for idx, data_tuple in self._get_data_iterator(keys, data, labels, weights):
                key, sample, label, weight = self._unpack_data_tuple(data_tuple)

                if partition < self._num_partitions - 1 and idx == fetch_next_partition_idx:
                    # TODO(#175) in case this blocks training
                    new_keys, new_data, new_labels, new_weights = self._get_data(
                        worker_id=worker_id, partition_id=partition + 1
                    )

                data_tuple = self._get_data_tuple(key, sample, label, weight)

                if data_tuple is not None:
                    yield data_tuple

            # this should mean we keep only two partitions in mem
            if partition < self._num_partitions - 1:
                del keys
                del data
                del labels
                del weights
                keys, data, labels, weights = new_keys, new_data, new_labels, new_weights
                del new_keys
                del new_data
                del new_labels
                del new_weights
                gc.collect()
