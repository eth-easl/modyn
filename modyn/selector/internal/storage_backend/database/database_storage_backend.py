import logging
import multiprocessing as mp
from typing import Any

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.storage_backend import AbstractStorageBackend

logger = logging.getLogger(__name__)


class DatabaseStorageBackend(AbstractStorageBackend):
    def __init__(self, pipeline_id: int, modyn_config: dict, maximum_keys_in_memory: int):
        super().__init__(pipeline_id, modyn_config, maximum_keys_in_memory)

    def persist_samples(
        self, seen_in_trigger_id: int, keys: list[int], timestamps: list[int], labels: list[int]
    ) -> dict[str, Any]:
        assert len(keys) == len(timestamps) and len(keys) == len(labels)

        log = {}
        swt = Stopwatch()

        # First persist the trigger which also creates the partition tables
        # This is done outside of subprocesses to avoid issues with duplicate table creation
        swt.start("trigger_creation")
        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.add_selector_state_metadata_trigger(self._pipeline_id, seen_in_trigger_id)
        log["trigger_creation_time"] = swt.stop()

        if self._disable_mt or (self._is_test and self._is_mac):
            swt.start("persist_samples_time")
            DatabaseStorageBackend._persist_samples_impl(
                keys, timestamps, labels, self._pipeline_id, self._modyn_config, seen_in_trigger_id
            )
            log["persist_samples_time"] = swt.stop()
        else:
            self._mt_persist_samples_impl(keys, timestamps, labels, seen_in_trigger_id, log)

        return log

    def _mt_persist_samples_impl(
        self, 
        keys: list[int],
        timestamps: list[int],
        labels: list[int],
        seen_in_trigger_id: int,
        log: dict
    ):
        swt = Stopwatch()
        samples_per_proc = int(len(keys) / self._insertion_threads)
        processes: list[mp.Process] = []
        swt.start("persist_samples_time")

        for i in range(self._insertion_threads):
            start_idx = i * samples_per_proc
            end_idx = start_idx + samples_per_proc if i < self._insertion_threads - 1 else len(keys)
            proc_keys = keys[start_idx:end_idx]
            proc_timestamps = timestamps[start_idx:end_idx]
            proc_labels = labels[start_idx:end_idx]

            if len(proc_keys) > 0:
                logger.debug(f"Starting persisting process for {len(proc_keys)} samples.")
                proc = mp.Process(
                    target=DatabaseStorageBackend._persist_samples_impl,
                    args=(
                        proc_keys,
                        proc_timestamps,
                        proc_labels,
                        self._pipeline_id,
                        self._modyn_config,
                        seen_in_trigger_id,
                    ),
                )
                proc.start()
                processes.append(proc)

        for proc in processes:
            proc.join()

        log["persist_samples_time"] = swt.stop()


    @staticmethod
    def _persist_samples_impl(
        keys: list[int],
        timestamps: list[int],
        labels: list[int],
        pipeline_id: int,
        modyn_config: dict,
        seen_in_trigger_id: int,
    ):
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

    def get_available_labels(self) -> list[int]:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            result = (
                database.session.query(SelectorStateMetadata.label)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id < self._next_trigger_id,
                    SelectorStateMetadata.seen_in_trigger_id >= self._next_trigger_id - self.tail_triggers - 1
                    if self.tail_triggers is not None
                    else True,
                )
                .distinct()
                .all()
            )
            available_labels = [result_tuple[0] for result_tuple in result]

        return available_labels