import logging
import multiprocessing as mp
import os
import platform
from abc import ABC, abstractmethod
from typing import Iterable, Optional

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata, Trigger, TriggerSample
from sqlalchemy import func

logger = logging.getLogger(__name__)


class AbstractSelectionStrategy(ABC):
    """This class is the base class for selection strategies.
    New selection strategies need to implement the
    `_on_trigger`, `_reset_state`, and `inform_data` methods.

    Args:
        config (dict): the configurations for the selector
        modyn_config (dict): the configurations for the modyn backend
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
        self._modyn_config = modyn_config
        self._pipeline_id = pipeline_id
        self._maximum_keys_in_memory = maximum_keys_in_memory
        self._insertion_threads = modyn_config["selector"]["insertion_threads"]

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
        training_samples: list[tuple[int, float]],
        modyn_config: dict,
        db_connection: Optional[MetadataDatabaseConnection],
    ) -> None:
        # In case we get passed a db_connection, we use that one (single-threaded execution)
        # and do not commit
        # Otherwise, we create a connection and commit and the end (MT-execution)
        # As our database connection only supports a context manager interface right now, we have to use these calls

        connection = (
            db_connection
            if db_connection is not None
            # pylint: disable-next=unnecessary-dunder-call
            else MetadataDatabaseConnection(modyn_config).__enter__()
        )

        connection.session.bulk_insert_mappings(
            TriggerSample,
            [
                {
                    "partition_id": partition_id,
                    "trigger_id": trigger_id,
                    "pipeline_id": pipeline_id,
                    "sample_key": key,
                    "sample_weight": weight,
                }
                for key, weight in training_samples
            ],
        )

        if db_connection is None:
            connection.session.commit()
            connection.__exit__(None, None, None)

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

        # If we run a test or disable MT, we cannot do multithreaded inserts
        # This is because sqlite does not allow us to insert while another connection is open
        # As iterating over on_trigger keeps an open connection, we have to commit after closing that connection
        # Hence we share a connection in that case and commit at the end.
        shared_conn = (
            # pylint: disable-next=unnecessary-dunder-call
            MetadataDatabaseConnection(self._modyn_config).__enter__()
            if self._is_test or self._disable_mt
            else None
        )

        partition: Optional[int] = None
        for partition, training_samples in enumerate(self._on_trigger()):
            logger.info(
                f"Strategy for pipeline {self._pipeline_id} returned batch of"
                + f" {len(training_samples)} samples for new trigger {trigger_id}."
            )

            total_keys_in_trigger += len(training_samples)

            if shared_conn is not None:
                AbstractSelectionStrategy._store_triggersamples_impl(
                    partition, trigger_id, self._pipeline_id, training_samples, self._modyn_config, shared_conn
                )

                continue

            samples_per_proc = int(len(training_samples) / self._insertion_threads)
            processes: list[mp.Process] = []

            for i in range(self._insertion_threads):
                start_idx = i * samples_per_proc
                end_idx = start_idx + samples_per_proc if i < self._insertion_threads - 1 else len(training_samples)
                proc_samples = training_samples[start_idx:end_idx]
                if len(proc_samples) > 0:
                    logger.debug(f"Starting trigger saving process for {len(proc_samples)} samples.")
                    proc = mp.Process(
                        target=AbstractSelectionStrategy._store_triggersamples_impl,
                        args=(partition, trigger_id, self._pipeline_id, proc_samples, self._modyn_config, None),
                    )
                    proc.start()
                    processes.append(proc)

            for proc in processes:
                proc.join()

        if shared_conn is not None:
            shared_conn.session.commit()
            shared_conn.__exit__(None, None, None)

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

        if self.reset_after_trigger:
            self._reset_state()

        self._next_trigger_id += 1
        return trigger_id, total_keys_in_trigger, num_partitions

    def get_trigger_partition_keys(self, trigger_id: int, partition_id: int) -> list[tuple[int, float]]:
        """
        Given a trigger id and partition id, returns a list of all keys in this partition

        Returns:
            list[tuple[int, float]]: list of training samples.
                Each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(TriggerSample.sample_key, TriggerSample.sample_weight)
                .filter(
                    TriggerSample.pipeline_id == self._pipeline_id,
                    TriggerSample.trigger_id == trigger_id,
                    TriggerSample.partition_id == partition_id,
                )
                .all()
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
