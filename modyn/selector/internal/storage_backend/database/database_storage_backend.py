import logging
import multiprocessing as mp
from typing import Any, Callable, Iterable, Optional

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.storage_backend import AbstractStorageBackend
from sqlalchemy import select

logger = logging.getLogger(__name__)


class DatabaseStorageBackend(AbstractStorageBackend):
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

        if self.insertion_threads == 1:
            swt.start("persist_samples_time")
            DatabaseStorageBackend._persist_samples_impl(
                keys, timestamps, labels, self._pipeline_id, self._modyn_config, seen_in_trigger_id
            )
            log["persist_samples_time"] = swt.stop()
        else:
            self._mt_persist_samples_impl(keys, timestamps, labels, seen_in_trigger_id, log)

        return log

    # pylint: disable=too-many-locals

    def _mt_persist_samples_impl(
        self, keys: list[int], timestamps: list[int], labels: list[int], seen_in_trigger_id: int, log: dict
    ) -> None:
        swt = Stopwatch()
        samples_per_proc = int(len(keys) / self.insertion_threads)
        processes: list[mp.Process] = []
        swt.start("persist_samples_time")

        for i in range(self.insertion_threads):
            start_idx = i * samples_per_proc
            end_idx = start_idx + samples_per_proc if i < self.insertion_threads - 1 else len(keys)
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

    def get_data_since_trigger(
        self, smallest_included_trigger_id: int
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        """Generator to get all samples seen since a certain trigger

        Returns:
            Iterable[tuple[list[int], dict[str, object]]]:
                Iterator over a tuple of a list of integers (maximum _maximum_keys_in_memory) and a log dict
        """
        additional_filter = (SelectorStateMetadata.seen_in_trigger_id >= smallest_included_trigger_id,)
        yield from self._get_pipeline_data(additional_filter)

    def get_trigger_data(self, trigger_id: int) -> Iterable[tuple[list[int], dict[str, object]]]:
        """Generator to get all samples seen during a certain trigger

        Returns:
            Iterable[tuple[list[int], dict[str, object]]]:
                Iterator over a tuple of a list of integers (maximum _maximum_keys_in_memory) and a log dict
        """
        additional_filter = (SelectorStateMetadata.seen_in_trigger_id == trigger_id,)
        yield from self._get_pipeline_data(additional_filter)

    def get_all_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        """Generator to get all samples seen

        Returns:
            Iterable[tuple[list[int], dict[str, object]]]:
                Iterator over a tuple of a list of integers (maximum _maximum_keys_in_memory) and a log dict
        """
        yield from self._get_pipeline_data(())

    def get_available_labels(self, next_trigger_id: int, tail_triggers: Optional[int] = None) -> list[int]:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            result = (
                database.session.query(SelectorStateMetadata.label)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id < next_trigger_id,
                    (
                        SelectorStateMetadata.seen_in_trigger_id >= next_trigger_id - tail_triggers - 1
                        if tail_triggers is not None
                        else True
                    ),
                )
                .distinct()
                .all()
            )
            available_labels = [result_tuple[0] for result_tuple in result]

        return available_labels

    def _get_pipeline_data(
        self,
        additional_filter: tuple,
        yield_per: Optional[int] = None,
        statement_modifier: Optional[Callable] = None,
        chunk_callback: Optional[Callable] = None,
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        """Internal generator to interact with database.

        Calling functions can extend the filtering by providing additional filter statements.
        yield_per defaults to self._maximum_keys_in_memory if not given.

        Returns:
            Iterable[tuple[list[int], dict[str, object]]]:
                Iterator over a tuple of a list of integers (maximum _maximum_keys_in_memory) and a log dict
        """
        yield_per = self._maximum_keys_in_memory if yield_per is None else yield_per
        filter_tuple = (SelectorStateMetadata.pipeline_id == self._pipeline_id,) + additional_filter

        stmt = select(SelectorStateMetadata.sample_key).execution_options(yield_per=yield_per).filter(*filter_tuple)

        if statement_modifier is not None:
            stmt = statement_modifier(stmt)

        yield from self._partitioned_execute_stmt(stmt, yield_per, chunk_callback)

    def _partitioned_execute_stmt(
        self, stmt: Any, yield_per: int, chunk_callback: Optional[Callable]
    ) -> Iterable[tuple[list[int], dict[str, object]]]:
        swt = Stopwatch()
        assert yield_per is not None

        stmt = stmt.execution_options(yield_per=yield_per)

        with MetadataDatabaseConnection(self._modyn_config) as database:
            swt.start("get_chunk")
            for chunk in database.session.execute(stmt).partitions():
                log = {"get_chunk_time": swt.stop()}

                if len(chunk) > 0:
                    if chunk_callback is not None:
                        chunk_callback(chunk)

                    yield [res[0] for res in chunk], log
                else:
                    yield [], log

                swt.start("get_chunk", overwrite=True)

    def _execute_on_session(self, session_callback: Callable) -> Any:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return session_callback(database.session)
