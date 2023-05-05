import logging
import multiprocessing as mp
import os
import platform
from abc import ABC, abstractmethod
from multiprocessing import shared_memory
from typing import Iterable, Optional

import numpy as np
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata, Trigger, TriggerPartition
from modyn.selector.internal.trigger_sample import TriggerSampleStorage
from sqlalchemy import func

logger = logging.getLogger(__name__)


class AbstractSelectionStrategy(ABC):
    """This class is the base class for selection strategies.
    New selection strategies need to implement the
    `_on_trigger`, `_reset_state`, and `inform_data` methods.

    Args:
        config (dict): the configurations for the selector
        modyn_config (dict): the configurations for the modyn module
    """

    def __init__(
        self,
        config: dict,
        modyn_config: dict,
        pipeline_id: int,
        maximum_keys_in_memory: int,
        required_configs: Optional[list[str]] = None,
    ):
        self._config = config

        if required_configs is None:
            required_configs = []  # Using [] as default is considered unsafe by pylint

        required_configs.extend(["limit", "reset_after_trigger"])
        for required_config in required_configs:
            if required_config not in self._config.keys():
                raise ValueError(f"{required_config} not given but required.")

        self.training_set_size_limit: int = config["limit"]
        self.has_limit = self.training_set_size_limit > 0
        self.reset_after_trigger: bool = config["reset_after_trigger"]
        if "tail_triggers" in config:
            self.tail_triggers = config["tail_triggers"]
            assert not (
                self.reset_after_trigger and self.tail_triggers > 0
            ), "Reset after trigger is equivalent to setting tail triggers to 0."
            assert not (
                not self.reset_after_trigger and self.tail_triggers == 0
            ), "Reset after trigger is equivalent to setting tail triggers to 0."
        else:
            if self.reset_after_trigger:
                self.tail_triggers = 0  # consider only the current trigger
            else:
                self.tail_triggers = -1  # consider every datapoint

        self._modyn_config = modyn_config
        self._pipeline_id = pipeline_id
        self._maximum_keys_in_memory = maximum_keys_in_memory
        self._insertion_threads = modyn_config["selector"]["insertion_threads"]
        self._requires_remote_computation = False

        self._is_test = "PYTEST_CURRENT_TEST" in os.environ
        self._is_mac = platform.system() == "Darwin"
        self._disable_mt = self._insertion_threads <= 0

        if self._maximum_keys_in_memory < 1:
            raise ValueError(f"Invalid setting for maximum_keys_in_memory: {self._maximum_keys_in_memory}")

        logger.info(f"Initializing selection strategy for pipeline {pipeline_id}.")

        with MetadataDatabaseConnection(self._modyn_config) as database:
            last_trigger_id = (
                database.session.query(func.max(Trigger.trigger_id))  # pylint: disable=not-callable
                .filter(Trigger.pipeline_id == self._pipeline_id)
                .scalar()
            )
            if last_trigger_id is None:
                logger.info(f"Did not find previous trigger id DB for pipeline {pipeline_id}, next trigger is 0.")
                self._next_trigger_id = 0
            else:
                logger.info(f"Last trigger in DB for pipeline {pipeline_id} was {last_trigger_id}.")
                self._next_trigger_id = last_trigger_id + 1

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

    @abstractmethod
    def _on_trigger(self) -> Iterable[list[tuple[int, float]]]:
        """
        Internal function. Defined by concrete strategy implementations. Calculates the next set of data to
        train on. Returns an iterator over lists, if next set of data consists of more than _maximum_keys_in_memory
        keys.

        Returns:
            Iterable[list[tuple[str, float]]]:
                Iterable over partitions. Each partition consists of a list of training samples.
                In each list, each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """
        raise NotImplementedError

    @abstractmethod
    def _reset_state(self) -> None:
        """Resets the internal state of the strategy, e.g., by clearing buffers."""
        raise NotImplementedError

    @abstractmethod
    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        """Informs the strategy of new data.

        Args:
            keys (list[str]): A list of keys of the data
            timestamps (list[int]): A list of timestamps of the data.
        """
        raise NotImplementedError

    @staticmethod
    def _store_triggersamples_impl(
        partition_id: int,
        trigger_id: int,
        pipeline_id: int,
        training_samples: np.ndarray,
        modyn_config: dict,
        insertion_id: int,
    ) -> None:
        TriggerSampleStorage(
            trigger_sample_directory=modyn_config["selector"]["trigger_sample_directory"],
        ).save_trigger_sample(
            pipeline_id=pipeline_id,
            trigger_id=trigger_id,
            partition_id=partition_id,
            trigger_samples=training_samples,
            insertion_id=insertion_id,
        )

    @staticmethod
    def _store_trigger_num_keys(
        modyn_config: dict,
        pipeline_id: int,
        trigger_id: int,
        partition_id: int,
        num_keys: int,
    ) -> None:
        with MetadataDatabaseConnection(modyn_config) as database:
            trigger_partition = TriggerPartition(
                pipeline_id=pipeline_id,
                trigger_id=trigger_id,
                partition_id=partition_id,
                num_keys=num_keys,
            )
            # TODO(#246): Maybe clean this up after some time.
            database.session.add(trigger_partition)
            database.session.commit()

    # pylint: disable=too-many-locals
    def trigger(self) -> tuple[int, int, int]:
        """
        Causes the strategy to compute the training set, and (if so configured) reset its internal state.

        Returns:
            tuple[int, int, int]: Trigger ID, how many keys are in the trigger, number of overall partitions
        """
        trigger_id = self._next_trigger_id
        total_keys_in_trigger = 0

        with MetadataDatabaseConnection(self._modyn_config) as database:
            trigger = Trigger(pipeline_id=self._pipeline_id, trigger_id=trigger_id)
            database.session.add(trigger)
            database.session.commit()

        partition_num_keys = {}
        partition: Optional[int] = None
        for partition, training_samples in enumerate(self._on_trigger()):
            logger.info(
                f"Strategy for pipeline {self._pipeline_id} returned batch of"
                + f" {len(training_samples)} samples for new trigger {trigger_id}."
            )

            partition_num_keys[partition] = len(training_samples)

            total_keys_in_trigger += len(training_samples)

            if (self._is_mac and self._is_test) or self._disable_mt:
                AbstractSelectionStrategy._store_triggersamples_impl(
                    partition,
                    trigger_id,
                    self._pipeline_id,
                    np.array(training_samples, dtype=np.dtype("i8,f8")),
                    self._modyn_config,
                    0,
                )
                continue

            samples_per_proc = int(len(training_samples) / self._insertion_threads)
            processes: list[mp.Process] = []

            for i in range(self._insertion_threads):
                start_idx = i * samples_per_proc
                end_idx = start_idx + samples_per_proc if i < self._insertion_threads - 1 else len(training_samples)
                proc_samples = np.array(training_samples[start_idx:end_idx], dtype=np.dtype("i8,f8"))
                if len(proc_samples) > 0:
                    shm = shared_memory.SharedMemory(
                        create=True,
                        size=proc_samples.nbytes,
                    )
                    shared_proc_samples: np.ndarray = np.ndarray(
                        proc_samples.shape, dtype=proc_samples.dtype, buffer=shm.buf
                    )
                    shared_proc_samples[:] = proc_samples  # This copies into the prepared numpy array
                    assert proc_samples.shape == shared_proc_samples.shape

                    logger.debug(f"Starting trigger saving process for {len(proc_samples)} samples.")
                    proc = mp.Process(
                        target=AbstractSelectionStrategy._store_triggersamples_impl,
                        args=(partition, trigger_id, self._pipeline_id, shared_proc_samples, self._modyn_config, i),
                    )
                    proc.start()
                    processes.append(proc)

            for proc in processes:
                proc.join()

        num_partitions = partition + 1 if partition is not None else 0

        # Update Trigger about number of partitions and keys
        with MetadataDatabaseConnection(self._modyn_config) as database:
            trigger = (
                database.session.query(Trigger)
                .filter(Trigger.pipeline_id == self._pipeline_id, Trigger.trigger_id == trigger_id)
                .first()
            )
            trigger.num_keys = total_keys_in_trigger
            trigger.num_partitions = num_partitions
            database.session.commit()

        # Insert all partition lengths into DB
        for partition, partition_keys in partition_num_keys.items():
            AbstractSelectionStrategy._store_trigger_num_keys(
                modyn_config=self._modyn_config,
                pipeline_id=self._pipeline_id,
                trigger_id=trigger_id,
                partition_id=partition,
                num_keys=partition_keys,
            )

        if self.reset_after_trigger:
            self._reset_state()

        self._next_trigger_id += 1
        return trigger_id, total_keys_in_trigger, num_partitions

    def get_trigger_partition_keys(
        self, trigger_id: int, partition_id: int, worker_id: int = -1, num_workers: int = -1
    ) -> list[tuple[int, float]]:
        """
        Given a trigger id and partition id, returns a list of all keys in this partition

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

        data = TriggerSampleStorage(
            self._trigger_sample_directory,
        ).get_trigger_samples(
            pipeline_id=self._pipeline_id,
            trigger_id=trigger_id,
            partition_id=partition_id,
            retrieval_worker_id=worker_id,
            total_retrieval_workers=num_workers,
            num_samples_trigger_partition=num_samples_trigger_partition,
        )

        assert len(data) <= self._maximum_keys_in_memory, "Chunking went wrong"

        return data

    @staticmethod
    def _persist_samples_impl(
        keys: list[int],
        timestamps: list[int],
        labels: list[int],
        pipeline_id: int,
        modyn_config: dict,
        seen_in_trigger_id: int,
    ) -> None:
        with MetadataDatabaseConnection(modyn_config) as database:
            database.session.bulk_insert_mappings(
                SelectorStateMetadata,
                [
                    {
                        "pipeline_id": pipeline_id,
                        "sample_key": key,
                        "timestamp": timestamp,
                        "label": label,
                        "seen_in_trigger_id": seen_in_trigger_id,
                    }
                    for key, timestamp, label in zip(keys, timestamps, labels)
                ],
            )
            database.session.commit()

    def _persist_samples(self, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        """Persists the data in the database.

        Args:
            keys (list[str]): A list of keys of the data
            timestamps (list[int]): A list of timestamps of the data.
            labels (list[int]): A list of labels of the data.
            database (MetadataDatabaseConnection): The database connection.
        """
        # TODO(#116): Right now we persist all datapoint into DB. We might want to
        # keep this partly in memory for performance.
        # Even if each sample is 64 byte and we see 2 million samples, it's just 128 MB of data in memory.
        # This also means that we have to clear this list on reset accordingly etc.
        assert len(keys) == len(timestamps) and len(keys) == len(labels)

        # First persist the trigger which also creates the partition tables
        #  This is done outside of subprocesses to avoid issues with duplicate table creation
        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.add_selector_state_metadata_trigger(self._pipeline_id, self._next_trigger_id)

        if self._disable_mt or (self._is_test and self._is_mac):
            AbstractSelectionStrategy._persist_samples_impl(
                keys, timestamps, labels, self._pipeline_id, self._modyn_config, self._next_trigger_id
            )
            return

        samples_per_proc = int(len(keys) / self._insertion_threads)
        processes: list[mp.Process] = []

        for i in range(self._insertion_threads):
            start_idx = i * samples_per_proc
            end_idx = start_idx + samples_per_proc if i < self._insertion_threads - 1 else len(keys)
            proc_keys = keys[start_idx:end_idx]
            proc_timestamps = timestamps[start_idx:end_idx]
            proc_labels = labels[start_idx:end_idx]
            if len(proc_keys) > 0:
                logger.debug(f"Starting persisting process for {len(proc_keys)} samples.")
                proc = mp.Process(
                    target=AbstractSelectionStrategy._persist_samples_impl,
                    args=(
                        proc_keys,
                        proc_timestamps,
                        proc_labels,
                        self._pipeline_id,
                        self._modyn_config,
                        self._next_trigger_id,
                    ),
                )
                proc.start()
                processes.append(proc)

        for proc in processes:
            proc.join()
