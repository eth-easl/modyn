import logging
import os
import platform
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

import numpy as np
from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.common.trigger_sample import ArrayWrapper, TriggerSampleStorage
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import Trigger, TriggerPartition
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

    # pylint: disable-next=too-many-branches
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

        # weighted optimization (with weights supplied by the selector) is quite unusual, so the default us false
        if "uses_weights" in config:
            self.uses_weights = config["uses_weights"]
        else:
            self.uses_weights = False

        if "tail_triggers" in config:
            self.tail_triggers = config["tail_triggers"]
            if self.tail_triggers < 0 or not isinstance(config["tail_triggers"], int):
                raise ValueError("Tail trigger must be an integer greater than 0")
            if (self.reset_after_trigger and self.tail_triggers > 0) or (
                (not self.reset_after_trigger) and self.tail_triggers == 0
            ):
                raise ValueError("Reset after trigger is equivalent to setting tail triggers to 0.")
        else:
            if self.reset_after_trigger:
                self.tail_triggers = 0  # consider only the current trigger
            else:
                self.tail_triggers = None  # consider every datapoint

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

        self._trigger_sample_directory = self._modyn_config["selector"]["trigger_sample_directory"]

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
        """
        Internal function. Defined by concrete strategy implementations. Calculates the next set of data to
        train on. Returns an iterator over lists, if next set of data consists of more than _maximum_keys_in_memory
        keys.

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
        """Resets the internal state of the strategy, e.g., by clearing buffers."""
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
        """Returns the list of all labels that could be returned in the latest trigger training set

        If the labels from the current "in progress" trigger should be included, first,
        we need to trigger, and then call this function, since this only includes data from the last finished trigger.
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

    # pylint: disable=too-many-locals,too-many-statements
    def trigger(self) -> tuple[int, int, int, dict[str, Any]]:
        """
        Causes the strategy to compute the training set, and (if so configured) reset its internal state.

        Returns:
            tuple[int, int, int]: Trigger ID, how many keys are in the trigger, number of overall partitions
        """
        # TODO(#276) Unify AbstractSelection Strategy and LocalDatasetWriter

        trigger_id = self._next_trigger_id
        total_keys_in_trigger = 0
        log: dict[str, Any] = {"trigger_partitions": []}
        swt = Stopwatch()

        swt.start("trigger_creation")
        with MetadataDatabaseConnection(self._modyn_config) as database:
            trigger = Trigger(pipeline_id=self._pipeline_id, trigger_id=trigger_id)
            database.session.add(trigger)
            database.session.commit()
        log["trigger_creation_time"] = swt.stop()

        partition_num_keys = {}
        partition: Optional[int] = None
        swt.start("on_trigger")

        for partition, (training_samples, partition_log) in enumerate(self._on_trigger()):
            overall_partition_log = {"partition_log": partition_log, "on_trigger_time": swt.stop("on_trigger")}

            logger.info(
                f"Strategy for pipeline {self._pipeline_id} returned batch of"
                + f" {len(training_samples)} samples for new trigger {trigger_id}."
            )

            partition_num_keys[partition] = len(training_samples)

            total_keys_in_trigger += len(training_samples)

            if (self._is_mac and self._is_test) or self._disable_mt:
                swt.start("store_triggersamples", overwrite=True)
                AbstractSelectionStrategy._store_triggersamples_impl(
                    partition,
                    trigger_id,
                    self._pipeline_id,
                    np.array(training_samples, dtype=np.dtype("i8,f8")),
                    [len(training_samples)],
                    self._modyn_config,
                )
                overall_partition_log["store_triggersamples_time"] = swt.stop()
                log["trigger_partitions"].append(overall_partition_log)
                swt.start("on_trigger", overwrite=True)
                continue

            swt.start("store_triggersamples", overwrite=True)
            swt.start("mt_prep", overwrite=True)
            samples_per_proc = int(len(training_samples) / self._insertion_threads)

            data_lengths = []

            overall_partition_log["mt_prep_time"] = swt.stop()
            swt.start("mt_finish", overwrite=True)

            if samples_per_proc > 0:
                data_lengths = [samples_per_proc] * (self._insertion_threads - 1)

            if sum(data_lengths) < len(training_samples):
                data_lengths.append(len(training_samples) - sum(data_lengths))

            AbstractSelectionStrategy._store_triggersamples_impl(
                partition,
                trigger_id,
                self._pipeline_id,
                np.array(training_samples, dtype=np.dtype("i8,f8")),
                data_lengths,
                self._modyn_config,
            )

            overall_partition_log["mt_finish_time"] = swt.stop()
            overall_partition_log["store_triggersamples_time"] = swt.stop("store_triggersamples")

            log["trigger_partitions"].append(overall_partition_log)
            swt.start("on_trigger", overwrite=True)

        swt.stop("on_trigger")
        num_partitions = partition + 1 if partition is not None else 0
        log["num_partitions"] = num_partitions
        log["num_keys"] = total_keys_in_trigger

        swt.start("db_update")
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

        log["db_update_time"] = swt.stop()

        if self.reset_after_trigger:
            swt.start("reset_state")
            self._reset_state()
            log["reset_state_time"] = swt.stop()

        self._next_trigger_id += 1
        return trigger_id, total_keys_in_trigger, num_partitions, log

    def get_trigger_partition_keys(
        self, trigger_id: int, partition_id: int, worker_id: int = -1, num_workers: int = -1
    ) -> ArrayWrapper:
        """
        Given a trigger id and partition id, returns an ArrayWrapper of all keys in this partition

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
