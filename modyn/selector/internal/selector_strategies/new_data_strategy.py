# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging
import random
from typing import Iterable

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from sqlalchemy import asc, select

logger = logging.getLogger(__name__)


class NewDataStrategy(AbstractSelectionStrategy):
    """
    This strategy always uses the latest data to train on.

    If we reset after trigger, we always use the data since the last trigger.
    If there is a limit, we choose a random subset from that data.
    This configuration can be used to either continously finetune
    or retrain from scratch on only new data.

    Without reset, we always output all data points since the pipeline has been started.
    If we do not have a limit, this can be used to retrain from scratch on all data.
    Finetuning does not really make sense here, unless you want to revisit all data all the time.
    If we have a limit, we use the "limit_reset" configuration option in the config dict to set a strategy.
    Currently we support "lastX" to train on the last LIMIT datapoints (ignoring triggers because we do not reset).
    We also support "sampleUAR" to sample a subset uniformly at random out of all data points.
    TODO(#125): In the future, we might want to support a sampling strategy that prioritizes newer data in some way.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        if self.has_limit and not self.reset_after_trigger and "limit_reset" not in config:
            raise ValueError("Please define how to deal with the limit without resets using the 'limit_reset' option.")

        if not (self.has_limit and not self.reset_after_trigger) and "limit_reset" in config:
            logger.warning("Since we do not have a limit and not reset, we ignore the 'limit_reset' setting.")

        self.supported_limit_reset_strategies = ["lastX", "sampleUAR"]
        if "limit_reset" in config:
            self.limit_reset_strategy = config["limit_reset"]

            if self.limit_reset_strategy not in self.supported_limit_reset_strategies:
                raise ValueError(f"Unsupported limit reset strategy: {self.limit_reset_strategy}")

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> dict[str, object]:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        swt = Stopwatch()
        swt.start("persist_samples")
        persist_log = self._persist_samples(keys, timestamps, labels)
        return {"total_persist_time": swt.stop(), "persist_log": persist_log}

    def _on_trigger(self) -> Iterable[tuple[list[tuple[int, float]], dict[str, object]]]:
        """
        Internal function. Defined by concrete strategy implementations. Calculates the next set of data to
        train on. Returns an iterator over lists, if next set of data consists of more than _maximum_keys_in_memory
        keys.

        Returns:
            Iterable[list[tuple[int, float]]]:
                Iterable over partitions. Each partition consists of a list of training samples.
                In each list, each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """
        if self.reset_after_trigger:
            assert self.tail_triggers == 0
            get_data_func = self._get_data_reset
        elif self.tail_triggers is not None and self.tail_triggers > 0:
            get_data_func = self._get_data_tail
        else:
            get_data_func = self._get_data_no_reset

        swt = Stopwatch()
        for samples, partition_log in get_data_func():
            swt.start("shuffle", overwrite=True)
            random.shuffle(samples)
            partition_log["shuffle_time"] = swt.stop()
            yield [(sample, 1.0) for sample in samples], partition_log

    def _get_data_reset(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        assert self.reset_after_trigger
        swt = Stopwatch()

        for samples, partition_log in self._get_current_trigger_data():
            # TODO(#179): this assumes limit < len(samples)
            if self.has_limit:
                swt.start("sample_time")
                samples = random.sample(samples, min(len(samples), self.training_set_size_limit))
                partition_log["sample_time"] = swt.stop()

            yield samples, partition_log

    def _get_data_tail(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        assert not self.reset_after_trigger and self.tail_triggers > 0
        swt = Stopwatch()

        for samples, partition_log in self._get_tail_triggers_data():
            # TODO(#179): this assumes limit < len(samples)
            if self.has_limit:
                swt.start("sample_time")
                samples = random.sample(samples, min(len(samples), self.training_set_size_limit))
                partition_log["sample_time"] = swt.stop()

            yield samples, partition_log

    def _get_data_no_reset(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        assert not self.reset_after_trigger
        swt = Stopwatch()
        for samples, partition_log in self._get_all_data():
            # TODO(#179): this assumes limit < len(samples)
            if self.has_limit:
                swt.start("handle_limit_time", overwrite=True)
                samples = self._handle_limit_no_reset(samples)
                partition_log["handle_limit_time"] = swt.stop()

            yield samples, partition_log

    def _handle_limit_no_reset(self, samples: list[int]) -> list[int]:
        assert self.limit_reset_strategy is not None

        if self.limit_reset_strategy == "lastX":
            return self._last_x_limit(samples)

        if self.limit_reset_strategy == "sampleUAR":
            return self._sample_uar(samples)

        raise NotImplementedError(f"Unsupport limit reset strategy: {self.limit_reset_strategy}")

    def _last_x_limit(self, samples: list[int]) -> list[int]:
        assert self.has_limit
        assert self.training_set_size_limit > 0

        return samples[-self.training_set_size_limit :]

    def _sample_uar(self, samples: list[int]) -> list[int]:
        assert self.has_limit
        assert self.training_set_size_limit > 0

        return random.sample(samples, min(len(samples), self.training_set_size_limit))

    def _get_current_trigger_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        """Returns all sample for current trigger

        Returns:
            list[int]: Keys of used samples
        """
        swt = Stopwatch()

        with MetadataDatabaseConnection(self._modyn_config) as database:
            stmt = (
                select(SelectorStateMetadata.sample_key)
                # Enables batching of results in chunks. See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._maximum_keys_in_memory)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id == self._next_trigger_id,
                )
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            swt.start("get_chunk")
            for chunk in database.session.execute(stmt).partitions():
                log = {"get_chunk_time": swt.stop()}

                if len(chunk) > 0:
                    yield [res[0] for res in chunk], log
                else:
                    yield [], log

                swt.start("get_chunk", overwrite=True)

    def _get_tail_triggers_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        """Returns all sample for current trigger

        Returns:
            list[int]: Keys of used samples
        """
        swt = Stopwatch()
        with MetadataDatabaseConnection(self._modyn_config) as database:
            stmt = (
                select(SelectorStateMetadata.sample_key)
                # Enables batching of results in chunks. See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._maximum_keys_in_memory)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id >= self._next_trigger_id - self.tail_triggers,
                )
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            swt.start("get_chunk")
            for chunk in database.session.execute(stmt).partitions():
                log = {"get_chunk_time": swt.stop()}

                if len(chunk) > 0:
                    yield [res[0] for res in chunk], log
                else:
                    yield [], log

                swt.start("get_chunk", overwrite=True)

    def _get_all_data(self) -> Iterable[tuple[list[int], dict[str, object]]]:
        """Returns all sample

        Returns:
            list[str]: Keys of used samples
        """
        swt = Stopwatch()
        with MetadataDatabaseConnection(self._modyn_config) as database:
            stmt = (
                select(SelectorStateMetadata.sample_key)
                # Enables batching of results in chunks. See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._maximum_keys_in_memory)
                .filter(SelectorStateMetadata.pipeline_id == self._pipeline_id)
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            swt.start("get_chunk")
            for chunk in database.session.execute(stmt).partitions():
                log = {"get_chunk_time": swt.stop()}

                if len(chunk) > 0:
                    yield [res[0] for res in chunk], log
                else:
                    yield [], log

                swt.start("get_chunk", overwrite=True)

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.
