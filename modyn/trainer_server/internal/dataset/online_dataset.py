import gc
import json
import logging
import os
import pathlib
import threading
from typing import Any, Callable, Generator, Optional, Tuple, Union

import grpc
from modyn.common.benchmark.stopwatch import Stopwatch
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
        prefetched_partitions: int,
        tokenizer: Optional[str],
        log_path: Optional[pathlib.Path],
    ):
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id
        self._training_id = training_id
        self._dataset_id = dataset_id
        self._first_call = True
        self._prefetched_partitions = prefetched_partitions

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
        self._log_path = log_path
        self._log: dict[str, Any] = {"partitions": {}}
        self._sw = Stopwatch()

        self._data_threads: dict[int, threading.Thread] = {}
        self._pref_started: dict[int, bool] = {}
        self._thread_data_container: dict[int, dict[str, Any]] = {}
        self._next_partition_to_fetch = 0

        if log_path is None:
            logger.warning("Did not provide log path for OnlineDataset - logging disabled.")

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

    def _get_data(self, data_container: dict, worker_id: int, partition_id: int) -> None:
        get_data_log = {}
        self._sw.start(f"GetKeysAndWeightsPart{partition_id}", overwrite=True)
        keys, weights = self._key_source.get_keys_and_weights(worker_id, partition_id)
        get_data_log["get_keys_and_weights"] = self._sw.stop(f"GetKeysAndWeightsPart{partition_id}")
        get_data_log["num_items"] = len(keys)

        self._info("Getting data from storage", worker_id)
        self._sw.start(f"GetDataPart{partition_id}", overwrite=True)
        data, labels = self._get_data_from_storage(keys)
        get_data_log["get_data"] = self._sw.stop(f"GetDataPart{partition_id}")

        self._log["partitions"][str(partition_id)] = get_data_log

        data_container["data"] = data
        data_container["keys"] = keys
        data_container["labels"] = labels
        data_container["weights"] = weights

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
        self._sw.start("transform", resume=True)
        # mypy complains here because _transform has unknown type, which is ok
        tranformed_sample = self._transform(sample)  # type: ignore
        self._sw.stop("transform")
        if self._uses_weights:
            return key, tranformed_sample, label, weight
        return key, tranformed_sample, label

    def end_of_trigger_cleaning(self) -> None:
        self._key_source.end_of_trigger_cleaning()

    def _persist_log(self, worker_id: int) -> None:
        if self._log_path is None:
            return

        if "PYTEST_CURRENT_TEST" in os.environ:
            json.dumps(self._log)  # Enforce serialization to catch issues
            return  # But don't actually store in tests

        log_file = f"{self._log_path / str(worker_id)}.log"
        self._log["transform"] = self._sw.measurements.get("transform", 0)
        self._log["wait_for_later_partitions"] = self._sw.measurements.get("wait_for_later_partitions", 0)
        self._log["wait_for_initial_partition"] = self._sw.measurements.get("wait_for_initial_partition", 0)

        with open(log_file, "w", encoding="utf-8") as logfile:
            json.dump(self._log, logfile)

    def _prefetch_partition(self, worker_id: int) -> None:
        if self._prefetched_partitions < 1 or self._next_partition_to_fetch >= self._num_partitions:
            return  # Prefetching disabled or nothing more to prefetch

        assert self._next_partition_to_fetch >= 0
        assert (
            self._next_partition_to_fetch not in self._data_threads
        ), f"Prefetching for partition {self._next_partition_to_fetch} has already been started"

        self._thread_data_container[self._next_partition_to_fetch]: dict[str, Any] = {}

        self._data_threads[self._next_partition_to_fetch] = threading.Thread(
            target=self._get_data,
            args=(self._thread_data_container[self._next_partition_to_fetch], worker_id, self._next_partition_to_fetch),
        )

        self._data_threads[self._next_partition_to_fetch].start()
        self._pref_started[self._next_partition_to_fetch] = True

        self._next_partition_to_fetch += 1

    def _wait_for_partition(
        self, worker_id: int, partition_id: int
    ) -> tuple[list[int], list[bytes], list[int], Optional[list[float]]]:
        container: dict[str, Any] = {}

        if self._prefetched_partitions < 1:
            # Prefetching disabled
            self._get_data(container, worker_id, partition_id)
        else:
            # Prefetching enabled
            assert self._pref_started[partition_id], f"Prefetching for partition {partition_id} has not been started"
            self._info(f"Joining thread for partition {partition_id}", worker_id)
            self._data_threads[partition_id].join()

            container = self._thread_data_container[partition_id]

        assert "data" in container and "labels" in container and "keys" in container and "weights" in container
        keys, data, labels, weights = (container["keys"], container["data"], container["labels"], container["weights"])
        container.clear()
        del container
        gc.collect()

        return keys, data, labels, weights

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements

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
            # Reinit logging, timetracking in this worker
            self._log = {"partitions": {}}
            self._sw = Stopwatch()

        # Always reinitialize these structures for prefetching (for multiple epochs)
        self._data_threads: dict[int, threading.Thread] = {}
        self._thread_data_container: dict[str, Any] = {}
        self._pref_started: dict[int, bool] = {}
        self._next_partition_to_fetch = 0

        assert self._transform is not None
        self._num_partitions = self._key_source.get_num_data_partitions()
        self._info(
            f"Total number of partitions will be {self._num_partitions}. Prefetch factor={self._prefetched_partitions}",
            worker_id,
        )
        self._log["num_partitions"] = self._num_partitions
        self._prefetched_partitions = min(self._prefetched_partitions, self._num_partitions)

        for partition in range(self._prefetched_partitions):
            self._prefetch_partition(worker_id)

        self._sw.start("wait_for_initial_partition", overwrite=True)
        keys, data, labels, weights = self._wait_for_partition(worker_id, 0)
        self._sw.stop("wait_for_initial_partition")

        for partition in range(self._num_partitions):
            self._persist_log(worker_id)
            num_samples_on_this_partition = len(keys)
            # We (arbitrarily) prefetch the next partition when we have seen 70% of the current partition
            fetch_next_partition_idx = int(num_samples_on_this_partition * 0.7)

            self._info(f"Train on partition {partition} ({num_samples_on_this_partition} samples)", worker_id)

            for idx, data_tuple in self._get_data_iterator(keys, data, labels, weights):
                key, sample, label, weight = self._unpack_data_tuple(data_tuple)

                if partition < self._num_partitions - 1 and idx == fetch_next_partition_idx:
                    self._prefetch_partition(worker_id)

                data_tuple = self._get_data_tuple(key, sample, label, weight)

                if data_tuple is not None:  # Can happen in PerClassDataset
                    yield data_tuple

            if partition < self._num_partitions - 1:
                del keys
                del data
                del labels
                del weights
                self._info(f"Partition {partition} completed, waiting for next partition", worker_id)
                self._sw.start("wait_for_later_partitions", resume=True)
                keys, data, labels, weights = self._wait_for_partition(worker_id, partition + 1)
                self._sw.stop("wait_for_later_partitions")
                gc.collect()

        self._persist_log(worker_id)
