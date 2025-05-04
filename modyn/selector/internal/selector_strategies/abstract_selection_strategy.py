import logging
import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from sqlalchemy import func

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.common.trigger_sample import ArrayWrapper, TriggerSampleStorage
from modyn.config.schema.pipeline import SelectionStrategy
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Trigger, TriggerPartition
from modyn.selector.internal.storage_backend import AbstractStorageBackend

logger = logging.getLogger(__name__)


class AbstractSelectionStrategy(ABC):
    """This class is the base class for selection strategies. New selection
    strategies need to implement the `_on_trigger`, `_reset_state`, and
    `inform_data` methods.

    Args:
        config (dict): the configurations for the selector
        modyn_config (dict): the configurations for the modyn module
    """

    # pylint: disable-next=too-many-branches
    def __init__(self, config: SelectionStrategy, modyn_config: dict, pipeline_id: int) -> None:
        self._config = config

        self.training_set_size_limit: int = config.limit
        self.has_limit = self.training_set_size_limit > 0

        # weighted optimization (with weights supplied by the selector) is quite unusual, so the default us false
        self.uses_weights = config.uses_weights

        self.tail_triggers = config.tail_triggers
        if self.tail_triggers is None:
            self.reset_after_trigger = False
        else:
            self.reset_after_trigger = self.tail_triggers == 0

        self._modyn_config = modyn_config
        self._pipeline_id = pipeline_id
        self._maximum_keys_in_memory = config.maximum_keys_in_memory

        logger.info(f"Initializing selection strategy for pipeline {pipeline_id}.")

        self._update_next_trigger_id()

        if self.has_limit and self.training_set_size_limit > self._maximum_keys_in_memory:
            # TODO(#179) Otherwise, we need to somehow sample over multiple not-in-memory partitions, which is a problem
            # Right now, we interpret the limit as a limit per partition
            # (this means a limit of 2 with 4 partitions with lead to 8 data points!)
            # This is also problematic since the limit now depends on the chunking. However, we need to think about
            # how to do this carefully
            raise ValueError(
                "We currently do not support a limit that is "
                "larger than the maximum amount of keys we may hold in memory."
            )

        self._trigger_sample_directory = self._modyn_config["selector"]["trigger_sample_directory"]
        self._storage_backend = self._init_storage_backend()

    def _update_next_trigger_id(self) -> None:
        tid = threading.get_native_id()
        pid = os.getpid()

        with MetadataDatabaseConnection(self._modyn_config) as database:
            last_trigger_id = (
                database.session.query(func.max(Trigger.trigger_id))  # pylint: disable=not-callable
                .filter(Trigger.pipeline_id == self._pipeline_id)
                .scalar()
            )
            if last_trigger_id is None:
                self._next_trigger_id = 0
                logger.info(
                    f"[{pid}][{tid}] Didn't find prev. trigger id for pipeline {self._pipeline_id}, next trigger = 0."
                )
            else:
                self._next_trigger_id = last_trigger_id + 1
                logger.info(
                    f"[{pid}][{tid}] Updating next trigger for pipeline {self._pipeline_id} to {self._next_trigger_id}."
                )

    @abstractmethod
    def _init_storage_backend(self) -> AbstractStorageBackend:
        """Initializes the storage backend."""
        raise NotImplementedError

    @property
    def maximum_keys_in_memory(self) -> int:
        return self._maximum_keys_in_memory

    @maximum_keys_in_memory.setter
    def maximum_keys_in_memory(self, value: int) -> None:
        self._maximum_keys_in_memory = value

        # Forward update of maximum keys in memory if needed
        if hasattr(self, "_storage_backend") and hasattr(self._storage_backend, "_maximum_keys_in_memory"):
            self._storage_backend._maximum_keys_in_memory = value

    @abstractmethod
    def _on_trigger(self) -> Iterable[tuple[list[tuple[int, float]], dict[str, Any]]]:
        """Internal function. Defined by concrete strategy implementations.
        Calculates the next set of data to train on. Returns an iterator over
        lists, if next set of data consists of more than
        _maximum_keys_in_memory keys.

        Returns:
            Iterable[tuple[list[tuple[int, float]], dict[str, Any]]]:
                Iterable over partitions. Each partition consists of a list of training samples.
                In each list, each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight. Each partition also has
                a log attached to it, the second element in the tuple
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_state(self) -> None:
        """Resets the internal state of the strategy, e.g., by clearing
        buffers."""
        raise NotImplementedError

    @abstractmethod
    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> dict[str, Any]:
        """Informs the strategy of new data.

        Args:
            keys (list[str]): A list of keys of the data
            timestamps (list[int]): A list of timestamps of the data.
            labels list[int]: A list of labels
        """
        raise NotImplementedError

    @abstractmethod
    def get_available_labels(self) -> list[int]:
        """Returns the list of all labels that could be returned in the latest
        trigger training set.

        If the labels from the current "in progress" trigger should be
        included, first, we need to trigger, and then call this
        function, since this only includes data from the last finished
        trigger.
        """
        raise NotImplementedError

    @staticmethod
    def _store_triggersamples_impl(
        partition_id: int,
        trigger_id: int,
        pipeline_id: int,
        training_samples: np.ndarray,
        data_lengths: list,
        modyn_config: dict,
    ) -> None:
        TriggerSampleStorage(
            trigger_sample_directory=modyn_config["selector"]["trigger_sample_directory"]
        ).save_trigger_samples(
            pipeline_id=pipeline_id,
            trigger_id=trigger_id,
            partition_id=partition_id,
            trigger_samples=training_samples,
            data_lengths=data_lengths,
        )

    @staticmethod
    def _store_trigger_num_keys(
        modyn_config: dict, pipeline_id: int, trigger_id: int, partition_id: int, num_keys: int
    ) -> None:
        with MetadataDatabaseConnection(modyn_config) as database:
            trigger_partition = TriggerPartition(
                pipeline_id=pipeline_id, trigger_id=trigger_id, partition_id=partition_id, num_keys=num_keys
            )
            # TODO(#246): Maybe clean this up after some time.
            database.session.add(trigger_partition)
            database.session.commit()

    # pylint: disable=too-many-locals
    @staticmethod
    def store_training_set(
        target_pipeline_id: int,
        target_trigger_id: int,
        modyn_config: dict,
        training_set_producer: Callable[[], Iterable[tuple[list[tuple[int, float]], dict[str, Any]]]],
        insertion_threads: int,
    ) -> tuple[int, int, dict[str, Any]]:
        """Store the training set, produced by the training_set_producer, as
        TriggerSampleStorage. Relevant metadata for the trigger is also stored
        in the metadata database.

        :param target_pipeline_id: the pipeline id the training set is
            associated with.
        :param target_trigger_id: the trigger id the training set is
            associated with.
        :param modyn_config: the modyn configuration.
        :param training_set_producer: a callable that returns
            partitioned training samples. The type is the same as the
            return type of the _on_trigger method.
        :param insertion_threads: how many threads are used to store. If
            bigger than 1, multiple threads are used to store the data.
        :return: total number of keys in the trigger, number of
            partitions, and a log.
        """
        # TODO(#276) Unify AbstractSelection Strategy and LocalDatasetWriter
        total_keys_in_trigger = 0
        log: dict[str, Any] = {"trigger_partitions": []}
        swt = Stopwatch()
        partition_num_keys = {}
        partition: int | None = None
        swt.start("on_trigger")

        for partition, (training_samples, partition_log) in enumerate(training_set_producer()):
            overall_partition_log = {"partition_log": partition_log, "on_trigger_time": swt.stop("on_trigger")}

            partition_num_keys[partition] = len(training_samples)

            total_keys_in_trigger += len(training_samples)

            swt.start("store_triggersamples", overwrite=True)
            if insertion_threads == 1:
                AbstractSelectionStrategy._store_triggersamples_impl(
                    partition,
                    target_trigger_id,
                    target_pipeline_id,
                    np.array(training_samples, dtype=np.dtype("i8,f8")),
                    [len(training_samples)],
                    modyn_config,
                )
            else:
                samples_per_proc = int(len(training_samples) / insertion_threads)
                data_lengths = []
                if samples_per_proc > 0:
                    data_lengths = [samples_per_proc] * (insertion_threads - 1)
                if sum(data_lengths) < len(training_samples):
                    data_lengths.append(len(training_samples) - sum(data_lengths))

                AbstractSelectionStrategy._store_triggersamples_impl(
                    partition,
                    target_trigger_id,
                    target_pipeline_id,
                    np.array(training_samples, dtype=np.dtype("i8,f8")),
                    data_lengths,
                    modyn_config,
                )
            overall_partition_log["store_triggersamples_time"] = swt.stop()

            log["trigger_partitions"].append(overall_partition_log)
            swt.start("on_trigger", overwrite=True)

        swt.stop("on_trigger")
        num_partitions = partition + 1 if partition is not None else 0
        log["num_partitions"] = num_partitions
        log["num_keys"] = total_keys_in_trigger

        swt.start("db_update")
        with MetadataDatabaseConnection(modyn_config) as database:
            trigger = Trigger(
                pipeline_id=target_pipeline_id,
                trigger_id=target_trigger_id,
                num_keys=total_keys_in_trigger,
                num_partitions=num_partitions,
            )
            database.session.add(trigger)
            database.session.commit()

        # Insert all partition lengths into DB
        for partition, partition_keys in partition_num_keys.items():
            AbstractSelectionStrategy._store_trigger_num_keys(
                modyn_config=modyn_config,
                pipeline_id=target_pipeline_id,
                trigger_id=target_trigger_id,
                partition_id=partition,
                num_keys=partition_keys,
            )

        log["db_update_time"] = swt.stop()
        return total_keys_in_trigger, num_partitions, log

    def trigger(self) -> tuple[int, int, int, dict[str, Any]]:
        """Causes the strategy to compute the training set, and (if so
        configured) reset its internal state.

        Returns:
            tuple[int, int, int]: Trigger ID, how many keys are in the trigger, number of overall partitions
        """
        total_keys_in_trigger, num_partitions, log = self.store_training_set(
            self._pipeline_id,
            self._next_trigger_id,
            self._modyn_config,
            self._on_trigger,
            insertion_threads=self._storage_backend.insertion_threads,
        )

        swt = Stopwatch()
        if self.reset_after_trigger:
            swt.start("reset_state")
            self._reset_state()
            log["reset_state_time"] = swt.stop()

        trigger_id = self._next_trigger_id
        self._next_trigger_id += 1
        return trigger_id, total_keys_in_trigger, num_partitions, log

    def get_trigger_partition_keys(
        self, trigger_id: int, partition_id: int, worker_id: int = -1, num_workers: int = -1
    ) -> ArrayWrapper:
        """Given a trigger id and partition id, returns an ArrayWrapper of all
        keys in this partition.

        Args:
            trigger_id (int): The trigger id
            partition_id (int): The partition id
            worker_id (int, optional): The worker id. Defaults to -1 meaning single threaded.
            num_workers (int, optional): The number of workers. Defaults to -1 meaning single threaded.

        Returns:
            list[tuple[int, float]]: list of training samples.
                Each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """

        with MetadataDatabaseConnection(self._modyn_config) as database:
            num_samples_trigger_partition = (
                database.session.query(TriggerPartition.num_keys)
                .filter(
                    TriggerPartition.pipeline_id == self._pipeline_id,
                    TriggerPartition.trigger_id == trigger_id,
                    TriggerPartition.partition_id == partition_id,
                )
                .first()
            )
            assert num_samples_trigger_partition is not None, f"Could not find TriggerPartition {partition_id} in DB"
            num_samples_trigger_partition = num_samples_trigger_partition[0]

        data = TriggerSampleStorage(self._trigger_sample_directory).get_trigger_samples(
            pipeline_id=self._pipeline_id,
            trigger_id=trigger_id,
            partition_id=partition_id,
            retrieval_worker_id=worker_id,
            total_retrieval_workers=num_workers,
            num_samples_trigger_partition=num_samples_trigger_partition,
        )

        assert len(data) <= self._maximum_keys_in_memory, "Chunking went wrong"

        return data
