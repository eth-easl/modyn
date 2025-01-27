from __future__ import annotations

import contextlib
import gc
import json
import logging
import os
import pathlib
import random
import threading
from collections.abc import Callable, Generator, Iterator
from typing import Any, cast

import grpc
from tenacity import (
    Retrying,
    after_log,
    before_log,
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.storage.internal.grpc.generated.storage_pb2 import (  # pylint: disable=no-name-in-module
    GetRequest,
    GetResponse,
)
from modyn.storage.internal.grpc.generated.storage_pb2_grpc import StorageStub
from modyn.trainer_server.internal.dataset.key_sources import (
    AbstractKeySource,
    SelectorKeySource,
)
from modyn.utils import (
    BYTES_PARSER_FUNC_NAME,
    deserialize_function,
    grpc_common_config,
    grpc_connection_established,
    instantiate_class,
)

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
        num_prefetched_partitions: int,
        parallel_prefetch_requests: int,
        shuffle: bool,
        tokenizer: str | None,
        log_path: pathlib.Path | None,
        include_labels: bool = True,
    ):
        self._include_labels = include_labels
        self._pipeline_id = pipeline_id
        self._trigger_id = trigger_id
        self._training_id = training_id
        self._dataset_id = dataset_id
        self._first_call = True
        self._num_prefetched_partitions = num_prefetched_partitions
        self._parallel_prefetch_requests = parallel_prefetch_requests

        self._bytes_parser = bytes_parser
        self._serialized_transforms = serialized_transforms
        self._storage_address = storage_address
        self._selector_address = selector_address
        self._transform_list: list[Callable] = []
        self._transform: Callable | None = None
        self._storagestub: StorageStub = None
        self._storage_channel: Any | None = None
        self._bytes_parser_function: Callable | None = None
        self._num_partitions = 0
        # the default key source is the Selector. Then it can be changed using change_key_source
        self._key_source: AbstractKeySource = SelectorKeySource(
            self._pipeline_id, self._trigger_id, self._selector_address
        )
        self._uses_weights: bool | None = None
        self._log_path = log_path
        self._log: dict[str, Any] = {"partitions": {}}
        self._log_lock: threading.Lock | None = None
        self._sw = Stopwatch()

        self._data_threads: dict[int, threading.Thread] = {}
        self._pref_started: dict[int, bool] = {}
        self._thread_data_container: dict[int, dict[str, Any]] = {}
        self._partition_locks: dict[int, threading.Lock] = {}
        self._partition_signals: dict[int, threading.Condition] = {}  # Should use the lock out of partition_locks
        self._partition_valid_until: dict[int, int] = {}
        self._partition_valid: dict[int, bool] = {}
        self._next_partition_to_fetch = 0
        self._launched_prefetches = 0
        self._start_prefetch_lock: threading.Lock | None = None
        self._shuffle = shuffle
        self._shuffled_partition_indices: list[int] = []

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

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_random_exponential(multiplier=1, min=2, max=60),
        before=before_log(logger, logging.ERROR),
        after=after_log(logger, logging.ERROR),
        reraise=True,
    )
    def _init_grpc(self, worker_id: int | None = None) -> None:  # pragma: no cover
        self._storage_channel = grpc.insecure_channel(self._storage_address, options=grpc_common_config())
        if grpc_connection_established(self._storage_channel):
            self._storagestub = StorageStub(self._storage_channel)
            return

        raise ConnectionError(
            f"[Worker {worker_id}]: Could not establish gRPC connection to storage at address {self._storage_address}."
        )

    def _silence_pil(self) -> None:  # pragma: no cover
        pil_logger = logging.getLogger("PIL")
        # by default, PIL on DEBUG spams the console
        pil_logger.setLevel(logging.INFO)

    def _info(self, msg: str, worker_id: int | None) -> None:  # pragma: no cover
        logger.info(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    def _debug(self, msg: str, worker_id: int | None) -> None:  # pragma: no cover
        logger.debug(f"[Training {self._training_id}][PL {self._pipeline_id}][Worker {worker_id}] {msg}")

    # pylint: disable=too-many-locals
    def _get_data_from_storage(
        self, selector_keys: list[int], worker_id: int | None = None
    ) -> Iterator[tuple[list[int], list[bytes], list[int] | None, int]]:
        processed_keys: set[int] | list[int] = []
        has_failed = False

        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_random_exponential(multiplier=1, min=2, max=60),
            reraise=True,
        ):
            with attempt:
                try:
                    new_keys: list[int]
                    new_samples: list[bytes]
                    response_time: int

                    # Select appropriate response type
                    response: GetResponse

                    # response: GetResponse # type: ignore[no-redef]
                    rpc_call = self._storagestub.Get

                    # Request setup
                    req = GetRequest(dataset_id=self._dataset_id, keys=selector_keys)
                    stopw = Stopwatch()
                    stopw.start("ResponseTime", overwrite=True)

                    for response in rpc_call(req):
                        response_time = stopw.stop("ResponseTime")
                        keys = list(response.keys)

                        if not has_failed:
                            assert isinstance(processed_keys, list)
                            processed_keys.extend(keys)
                            if not self._include_labels:
                                yield keys, list(response.samples), None, response_time
                            else:
                                yield keys, list(response.samples), list(response.labels), response_time
                        else:  # Handle failure and deduplication
                            assert isinstance(processed_keys, set)
                            new_keys = [key for key in keys if key not in processed_keys]
                            new_samples = [
                                sample for key, sample in zip(keys, response.samples) if key not in processed_keys
                            ]
                            processed_keys.update(keys)

                            if self._include_labels:
                                yield new_keys, new_samples, None, response_time
                            else:
                                new_labels = [
                                    label for key, label in zip(keys, response.labels) if key not in processed_keys
                                ]
                                yield new_keys, new_samples, new_labels, response_time

                        stopw.start("ResponseTime", overwrite=True)
                except grpc.RpcError as e:
                    has_failed = True
                    # Convert processed keys to set on first failure
                    processed_keys = set(processed_keys) if isinstance(processed_keys, list) else processed_keys
                    self._info(
                        "gRPC error occurred, processed_keys = " + f"{processed_keys}\n{e.code()} - {e.details()}",
                        worker_id,
                    )
                    self._info(f"Stringified exception: {str(e)}", worker_id)
                    self._info(
                        f"Error occurred while asking {self._dataset_id} for keys:\n{selector_keys}",
                        worker_id,
                    )
                    self._init_grpc(worker_id=worker_id)
                    raise e

    # pylint: disable=too-many-locals

    def _get_data(
        self,
        data_container: dict,
        worker_id: int,
        partition_id: int,
        partition_valid: dict | None,
        partition_valid_until: dict | None,
        partition_locks: dict | None,
        partition_signals: dict | None,
        callback: Callable | None,
        shuffled_partition_id: int | None,
    ) -> None:
        get_data_log = {}
        self._sw.start(f"GetKeysAndWeightsPart{partition_id}", overwrite=True)
        keys, weights = self._key_source.get_keys_and_weights(
            worker_id,
            (shuffled_partition_id if shuffled_partition_id is not None else partition_id),
        )
        get_data_log["get_keys_and_weights"] = self._sw.stop(f"GetKeysAndWeightsPart{partition_id}")
        get_data_log["num_items"] = len(keys)

        self._info("Getting data from storage", worker_id)
        self._sw.start(f"GetDataPart{partition_id}", overwrite=True)
        all_response_times = []
        key_weight_map = {key: weights[idx] for idx, key in enumerate(keys)} if weights is not None else None
        if not self._include_labels:
            for data_tuple_gen in self._get_data_from_storage(keys, worker_id=worker_id):
                stor_keys, data, _, response_time = data_tuple_gen

                all_response_times.append(response_time)
                num_items = len(stor_keys)
                with partition_locks[partition_id] if partition_locks is not None else contextlib.suppress():
                    data_container["data"].extend(data)
                    data_container["keys"].extend(stor_keys)
                    data_container["weights"].extend(
                        [cast(float | None, key_weight_map[key]) for key in stor_keys]
                        if key_weight_map is not None
                        else [None for _ in range(len(stor_keys))]
                    )
                    if partition_valid_until is not None:
                        partition_valid_until[partition_id] += num_items

                if partition_signals is not None:
                    with partition_signals[partition_id]:
                        partition_signals[partition_id].notify_all()
        else:
            for data_tuple in self._get_data_from_storage(keys, worker_id=worker_id):
                stor_keys, data, labels, response_time = data_tuple

                all_response_times.append(response_time)
                num_items = len(stor_keys)
                with partition_locks[partition_id] if partition_locks is not None else contextlib.suppress():
                    data_container["data"].extend(data)
                    data_container["keys"].extend(stor_keys)
                    data_container["labels"].extend(labels)
                    data_container["weights"].extend(
                        [cast(float | None, key_weight_map[key]) for key in stor_keys]
                        if key_weight_map is not None
                        else [None for _ in range(len(stor_keys))]
                    )
                    if partition_valid_until is not None:
                        partition_valid_until[partition_id] += num_items

                if partition_signals is not None:
                    with partition_signals[partition_id]:
                        partition_signals[partition_id].notify_all()

        get_data_log["get_data"] = self._sw.stop(f"GetDataPart{partition_id}")
        get_data_log["response_times"] = all_response_times
        assert self._log_lock is not None

        with self._log_lock:
            self._log["partitions"][str(partition_id)] = get_data_log

        if partition_locks is not None and partition_valid is not None:
            with partition_locks[partition_id]:
                partition_valid[partition_id] = True

        if callback is not None:
            callback()

    def _get_transformed_data_tuple(
        self, key: int, sample: memoryview, label: int | None = None, weight: float | None = None
    ) -> tuple | None:
        assert self._uses_weights is not None
        self._sw.start("transform", resume=True)
        transformed_sample = self._transform(sample)  # type: ignore
        self._sw.stop("transform")

        if not self._include_labels:
            if self._uses_weights:
                return key, transformed_sample, weight
            return key, transformed_sample

        # Non-include_labels case with labels
        if self._uses_weights:
            return key, transformed_sample, label, weight
        return key, transformed_sample, label

    def end_of_trigger_cleaning(self) -> None:
        self._key_source.end_of_trigger_cleaning()

    def _persist_log(self, worker_id: int) -> None:
        if self._log_path is None:
            return

        assert self._log_lock is not None

        with self._log_lock:
            if "PYTEST_CURRENT_TEST" in os.environ:
                json.dumps(self._log)  # Enforce serialization to catch issues
                return  # But don't actually store in tests

            log_file = f"{self._log_path / str(worker_id)}.log"
            self._log["transform"] = self._sw.measurements.get("transform", 0)
            self._log["wait_for_later_partitions"] = self._sw.measurements.get("wait_for_later_partitions", 0)
            self._log["wait_for_initial_partition"] = self._sw.measurements.get("wait_for_initial_partition", 0)

            with open(log_file, "w", encoding="utf-8") as logfile:
                json.dump(self._log, logfile)

    def _clear_partition(self, partition_id: int) -> None:
        with self._partition_locks[partition_id] if self._partition_locks is not None else contextlib.suppress():
            self._partition_valid[partition_id] = False
            self._partition_valid_until[partition_id] = -1
            del self._thread_data_container[partition_id]

        if "PYTEST_CURRENT_TEST" not in os.environ:
            gc.collect()

    def _prefetch_partition(self, worker_id: int, maybe_continue: bool = False) -> None:
        assert self._start_prefetch_lock is not None
        with self._start_prefetch_lock:
            if self._num_prefetched_partitions < 1 or self._next_partition_to_fetch >= self._num_partitions:
                return  # Prefetching disabled or nothing more to prefetch

            if maybe_continue and self._launched_prefetches >= self._num_prefetched_partitions:
                return  # Two callbacks started to prefetch basically at the same time

            if maybe_continue:
                # Do this as early as possible to avoid running into the "problem" above frequently
                self._launched_prefetches += 1
            assert self._next_partition_to_fetch >= 0
            assert (
                self._next_partition_to_fetch not in self._data_threads
            ), f"Prefetching for partition {self._next_partition_to_fetch} has already been started"
            if self._include_labels:
                self._thread_data_container[self._next_partition_to_fetch] = {
                    "data": [],
                    "keys": [],
                    "labels": [],
                    "weights": [],
                }
            else:
                self._thread_data_container[self._next_partition_to_fetch] = {
                    "data": [],
                    "keys": [],
                    "weights": [],
                }
            self._partition_valid[self._next_partition_to_fetch] = False
            self._partition_valid_until[self._next_partition_to_fetch] = -1
            self._partition_locks[self._next_partition_to_fetch] = threading.Lock()
            self._partition_signals[self._next_partition_to_fetch] = threading.Condition(
                self._partition_locks[self._next_partition_to_fetch]
            )

            callback = None
            if maybe_continue:

                def callback_func() -> None:
                    self._info("Prefetch callback called.", worker_id)

                    # It might be that between the check and the actual launch
                    # We start another launch
                    # We catch this with the lock within _prefetch_partition
                    if self._launched_prefetches < self._num_prefetched_partitions:
                        self._info(
                            f"Only {self._launched_prefetches} out of {self._num_prefetched_partitions}"
                            + " partitions have been fetched, issuing another request.",
                            worker_id,
                        )
                        self._prefetch_partition(worker_id, True)
                    else:
                        self._info("Not issuing another request.", worker_id)

                callback = callback_func

            # We implement shuffling on a partition level by mapping everything to increasing indices but actually load
            # different partition data.
            shuffle_partition_id = (
                self._shuffled_partition_indices[self._next_partition_to_fetch] if self._shuffle else None
            )
            self._data_threads[self._next_partition_to_fetch] = threading.Thread(
                target=self._get_data,
                args=(
                    self._thread_data_container[self._next_partition_to_fetch],
                    worker_id,
                    self._next_partition_to_fetch,
                    self._partition_valid,
                    self._partition_valid_until,
                    self._partition_locks,
                    self._partition_signals,
                    callback,
                    shuffle_partition_id,
                ),
            )

            self._data_threads[self._next_partition_to_fetch].start()
            self._pref_started[self._next_partition_to_fetch] = True

            self._next_partition_to_fetch += 1

    def _fetch_partition_noprefetch(
        self, worker_id: int, partition_id: int
    ) -> Iterator[tuple[int, memoryview, int, float | None]] | Iterator[tuple[int, memoryview, float | None]]:
        assert self._num_prefetched_partitions < 1
        container: dict[str, Any] = {
            "data": [],
            "keys": [],
            "weights": [],
        }
        if self._include_labels:
            container["labels"] = []

        shuffle_partition_id = self._shuffled_partition_indices[partition_id] if self._shuffle else None

        self._get_data(
            container,
            worker_id,
            partition_id,
            None,
            None,
            None,
            None,
            None,
            shuffle_partition_id,
        )

        assert "data" in container and "keys" in container and "weights" in container
        if self._include_labels:
            assert "labels" in container

        if self._shuffle:
            self._shuffle_partition(partition_id, worker_id, container=container)

        for idx in range(len(container["keys"])):
            if not self._include_labels:
                yield (
                    container["keys"][idx],
                    memoryview(container["data"][idx]),
                    container["weights"][idx],
                )
            else:
                yield (
                    container["keys"][idx],
                    memoryview(container["data"][idx]),
                    container["labels"][idx],
                    container["weights"][idx],
                )

    def _is_partition_fetched(self, partition_id: int) -> bool:
        if partition_id not in self._partition_locks or partition_id not in self._partition_valid:
            return False

        with self._partition_locks[partition_id]:
            return self._partition_valid[partition_id]

    def _partition_max_index(self, partition_id: int) -> int:
        with self._partition_locks[partition_id]:
            return self._partition_valid_until[partition_id]

    def _get_partition_data(
        self, last_idx: int, max_idx: int, partition_id: int
    ) -> Iterator[tuple[int, memoryview, int, float | None]] | Iterator[tuple[int, memoryview, float | None]]:
        if not self._include_labels:
            for idx in range(last_idx + 1, max_idx + 1):
                yield (
                    self._thread_data_container[partition_id]["keys"][idx],
                    memoryview(self._thread_data_container[partition_id]["data"][idx]),
                    self._thread_data_container[partition_id]["weights"][idx],
                )
        else:
            for idx in range(last_idx + 1, max_idx + 1):
                yield (
                    self._thread_data_container[partition_id]["keys"][idx],
                    memoryview(self._thread_data_container[partition_id]["data"][idx]),
                    self._thread_data_container[partition_id]["labels"][idx],
                    self._thread_data_container[partition_id]["weights"][idx],
                )

    def _wait_for_new_partition_data(self, partition_id: int) -> None:
        with self._partition_signals[partition_id]:
            self._partition_signals[partition_id].wait(1)  # In case we do not get woken up, we at most waste a second

    def _shuffle_partition(self, partition_id: int, worker_id: int, container: dict | None = None) -> None:
        assert container is not None or self._is_partition_fetched(partition_id)
        assert self._shuffle

        container = container if container is not None else self._thread_data_container[partition_id]

        self._info(f"Shuffling partition {partition_id}", worker_id)

        data_length = len(container["data"])
        indices = list(range(data_length))
        random.shuffle(indices)

        container["data"] = [container["data"][i] for i in indices]
        container["keys"] = [container["keys"][i] for i in indices]
        container["weights"] = [container["weights"][i] for i in indices]

        if self._include_labels:
            container["labels"] = [container["labels"][i] for i in indices]

        self._info(f"Shuffled partition {partition_id}", worker_id)

    def prefetched_partition_generator(
        self, worker_id: int, partition_id: int
    ) -> Iterator[tuple[int, memoryview, int, float | None]] | Iterator[tuple[int, memoryview, float | None]]:
        last_idx = -1
        if not self._shuffle:
            # If we do not shuffle, we can emit data as soon as it streamed over
            # Otherwise, we have to wait for the transfer to finish
            while not self._is_partition_fetched(partition_id):
                max_idx = self._partition_max_index(partition_id)
                if max_idx <= last_idx:  # No new data
                    self._wait_for_new_partition_data(partition_id)
                yield from self._get_partition_data(last_idx, max_idx, partition_id)
                last_idx = max_idx
        else:
            while not self._is_partition_fetched(partition_id):
                self._wait_for_new_partition_data(partition_id)
            self._shuffle_partition(partition_id, worker_id)

        # Yield potential remaining data (when not shuffling) or all data (when shuffling)
        self._info(f"Joining thread for partition {partition_id}", worker_id)
        self._data_threads[partition_id].join()
        self._info(f"Thread for partition {partition_id} joined", worker_id)
        max_idx = self._partition_max_index(partition_id)
        yield from self._get_partition_data(last_idx, max_idx, partition_id)
        self._info(f"Clearing partition {partition_id}", worker_id)
        self._clear_partition(partition_id)

    def start_prefetching(self, worker_id: int) -> None:
        if self._num_prefetched_partitions < 1:
            # No prefetching at all
            return

        if self._num_prefetched_partitions <= self._parallel_prefetch_requests:
            # We can emit prefetching requests once and be done with it
            for _ in range(self._num_prefetched_partitions):
                self._prefetch_partition(worker_id, False)

            return

        # We have to respect the limit of parallel requests
        for _ in range(self._parallel_prefetch_requests):
            self._prefetch_partition(worker_id, True)

    def all_partition_generator(
        self, worker_id: int
    ) -> Iterator[tuple[int, memoryview, int, float | None]] | Iterator[tuple[int, memoryview, float | None]]:
        self.start_prefetching(worker_id)

        for partition_id in range(self._num_partitions):
            self._persist_log(worker_id)
            if self._num_prefetched_partitions > 0:
                if partition_id < self._num_partitions - 1:
                    # As we consume one partition, prefetch exactly one more partition
                    self._prefetch_partition(worker_id, False)

                yield from self.prefetched_partition_generator(worker_id, partition_id)

            else:
                yield from self._fetch_partition_noprefetch(worker_id, partition_id)

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
            self._init_grpc(worker_id)
            self._key_source.init_worker()
            self._uses_weights = self._key_source.uses_weights()
            self._silence_pil()
            self._debug("gRPC initialized.", worker_id)
            # Reinit logging, timetracking in this worker
            self._log = {"partitions": {}}
            self._sw = Stopwatch()
            self._start_prefetch_lock = threading.Lock()
            self._log_lock = threading.Lock()

        # Always reinitialize these structures for prefetching (for multiple epochs)
        self._data_threads = {}
        self._thread_data_container = {}
        self._pref_started = {}
        self._next_partition_to_fetch = 0
        self._partition_locks = {}
        self._partition_valid_until = {}
        self._partition_valid = {}
        self._partition_signals = {}

        assert self._transform is not None
        self._num_partitions = self._key_source.get_num_data_partitions()

        if self._shuffle:
            self._shuffled_partition_indices = list(range(0, self._num_partitions))
            random.shuffle(self._shuffled_partition_indices)
            self._info(
                f"Shuffled partitions into random order: {self._shuffled_partition_indices}",
                worker_id,
            )

        self._info(
            f"Total number of partitions will be {self._num_partitions}.\n"
            + f"Parallel prefetch requests = {self._parallel_prefetch_requests}\n"
            + f"Num prefetched partitions = {self._num_prefetched_partitions}",
            worker_id,
        )
        assert self._log_lock is not None
        with self._log_lock:
            self._log["num_partitions"] = self._num_partitions
        self._num_prefetched_partitions = min(self._num_prefetched_partitions, self._num_partitions)

        for data_tuple in self.all_partition_generator(worker_id):
            if (transformed_tuple := self._get_transformed_data_tuple(*data_tuple)) is not None:
                yield transformed_tuple

        self._persist_log(worker_id)
