# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging
import random
from math import isclose
from typing import Iterable, Iterator

from modyn.common.benchmark.stopwatch import Stopwatch
from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from sqlalchemy import asc, exc, func, select, update

logger = logging.getLogger(__name__)


class FreshnessSamplingStrategy(AbstractSelectionStrategy):
    """
    This class selects data from a mixture of used and unused data.
    We can set a ratio that defines how much data in the training set per trigger should be from previously unused data (in all previous triggers).

    The first trigger will always use only fresh data (up to the limit, if there is one).
    The subsequent triggers will sample a dataset that reflects the ratio of used/unused data (if data came during but was not used in a previous trigger, we still handle it as unused).
    We have to respect both the ratio and the limit (if there is one) and build up the dataset on trigger accordingly.

    It cannot be used with reset, because we need to keep state over multiple triggers.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(
            config, modyn_config, pipeline_id, maximum_keys_in_memory, required_configs=["unused_data_ratio"]
        )
        assert self.tail_triggers is None or self.tail_triggers == 0, "Tail triggers not supported for this strategy."
        self.unused_data_ratio = self._config["unused_data_ratio"]
        self._is_first_trigger = True

        if self.unused_data_ratio < 1 or self.unused_data_ratio > 99:
            raise ValueError(
                f"Invalid unused data ratio: {self.unused_data_ratio}. We need at least 1% fresh data (otherwise we would always train on the data from first trigger) and at maximum 99% fresh data (otherwise please use NewDataStrategy+reset)."
            )

        if self.reset_after_trigger:
            raise ValueError(
                "FreshnessSamplingStrategy cannot reset state after trigger, because then no old data would be available to sample from."
            )

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> dict[str, object]:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        swt = Stopwatch()
        swt.start("persist_samples")
        persist_log = self._persist_samples(keys, timestamps, labels)
        return {"total_persist_time": swt.stop(), "persist_log": persist_log}

    def _on_trigger(self) -> Iterable[tuple[list[tuple[int, float]], dict[str, object]]]:
        """
        Internal function. Calculates the next set of data to
        train on.

        Returns:
            list(tuple(str, float)): Each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """

        # TODO(#116): right now this is an offline implementation. we might switch to an online
        # implementation where we don't calculate everything on trigger. This depends on what
        # we hold in memory.

        if self._is_first_trigger:
            get_data_func = self._get_first_trigger_data
        else:
            get_data_func = self._get_trigger_data

        for samples in get_data_func():
            self._mark_used(samples)
            random.shuffle(samples)

            # Add logging here when required.
            yield [(sample, 1.0) for sample in samples], {}

    def _get_first_trigger_data(self) -> Iterable[list[int]]:
        assert self._is_first_trigger
        self._is_first_trigger = False

        for samples in self._get_all_unused_data():
            # TODO(#179): this assumes limit < len(samples)
            if self.has_limit and self.training_set_size_limit < len(samples):
                samples = random.sample(samples, self.training_set_size_limit)

            yield samples

    def _get_trigger_data(self) -> Iterable[list[int]]:
        assert not self._is_first_trigger
        count_unused_samples = self._get_count_of_data(False)
        count_used_samples = self._get_count_of_data(True)

        num_unused_samples, num_used_samples = self._calc_num_samples_no_limit(count_unused_samples, count_used_samples)

        if self.has_limit and (num_unused_samples + num_used_samples) > self.training_set_size_limit:
            num_unused_samples, num_used_samples = self._calc_num_samples_limit(
                count_unused_samples, count_used_samples
            )

        # Idea: We issue a windowed query for unused and used samples with _maximum_keys_in_memory / 2 as the window size each.
        # We then always concatenate the two windows of the subsample to create a window of _maximum_keys_in_memory which we yield

        unused_generator = self._get_data_sample(num_unused_samples, False)
        used_generator = self._get_data_sample(num_used_samples, True)

        next_unused_sample: list[int] = next(unused_generator, [])
        next_used_sample: list[int] = next(used_generator, [])

        while len(next_unused_sample) > 0 or len(next_used_sample) > 0:
            yield next_unused_sample + next_used_sample
            next_unused_sample = next(unused_generator, [])
            next_used_sample = next(used_generator, [])

    def _calc_num_samples_no_limit(self, total_unused_samples: int, total_used_samples: int) -> tuple[int, int]:
        # For both the used and unused samples, we calculate how many samples we could have at maximum where the used/unused samples make up the required fraction

        maximum_samples_unused = int(total_unused_samples / (float(self.unused_data_ratio) / 100.0))
        maximum_samples_used = int(total_used_samples / (float(100 - self.unused_data_ratio) / 100.0))

        if maximum_samples_unused > maximum_samples_used:
            total_samples = maximum_samples_used
            num_used_samples = total_used_samples
            num_unused_samples = total_samples - num_used_samples
        else:
            total_samples = maximum_samples_unused
            num_unused_samples = total_unused_samples
            num_used_samples = total_samples - num_unused_samples

        assert isclose(num_unused_samples / total_samples, float(self.unused_data_ratio) / 100.0, abs_tol=0.5)
        assert num_used_samples <= total_used_samples
        assert num_unused_samples <= total_unused_samples

        return num_unused_samples, num_used_samples

    def _calc_num_samples_limit(self, total_unused_samples: int, total_used_samples: int) -> tuple[int, int]:
        assert self.has_limit

        # This function has the assumption that we have enough data points available to fulfill the limit
        # This is why _get_trigger_data calls the no limit function first
        num_unused_samples = int(self.training_set_size_limit * (float(self.unused_data_ratio) / 100.0))
        num_used_samples = self.training_set_size_limit - num_unused_samples

        assert num_unused_samples <= total_unused_samples
        assert num_used_samples <= total_used_samples

        return num_unused_samples, num_used_samples

    def _get_data_sample(self, sample_size: int, used: bool) -> Iterator[list[int]]:
        """Returns sample of data. Returns ins batches of  self._maximum_keys_in_memory / 2

        Returns:
            list[str]: Keys of used samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            stmt = (
                select(SelectorStateMetadata.sample_key, SelectorStateMetadata.used)
                # Enables batching of results in chunks. See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=max(int(self._maximum_keys_in_memory / 2), 1))
                .filter(SelectorStateMetadata.pipeline_id == self._pipeline_id, SelectorStateMetadata.used == used)
                .order_by(func.random())  # pylint: disable=not-callable
                .limit(sample_size)
            )

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    keys, used_data = zip(*chunk)
                    if used:
                        assert all(used_data), "Queried used data, but got unused data."
                    else:
                        assert not any(used_data), "Queried unused data, but got used data."

                else:
                    keys = []

                yield list(keys)

    def _get_all_unused_data(self) -> Iterator[list[int]]:
        """Returns all unused samples

        Returns:
            list[str]: Keys of unused samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            stmt = (
                select(SelectorStateMetadata.sample_key, SelectorStateMetadata.used)
                # Enables batching of results in chunks. See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._maximum_keys_in_memory)
                .filter(SelectorStateMetadata.pipeline_id == self._pipeline_id, SelectorStateMetadata.used == False)
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    keys, used = zip(*chunk)
                    assert not any(used), "Queried unused data, but got used data."
                else:
                    keys, used = [], []

                yield list(keys)

    def _get_count_of_data(self, used: bool) -> int:
        """Returns all unused samples

        Returns:
            list[str]: Keys of unused samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return (
                database.session.query(SelectorStateMetadata.sample_key)
                # TODO(#182): Index on used?
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id, SelectorStateMetadata.used == used
                ).count()
            )

    def _mark_used(self, keys: list[int]) -> None:
        """Sets samples to used"""
        if len(keys) == 0:
            return

        with MetadataDatabaseConnection(self._modyn_config) as database:
            try:
                stmt = update(SelectorStateMetadata).where(SelectorStateMetadata.sample_key.in_(keys)).values(used=True)
                database.session.execute(stmt)
                database.session.commit()
            except exc.SQLAlchemyError as exception:
                logger.error(f"Could not set metadata: {exception}")
                database.session.rollback()

    def _reset_state(self) -> None:
        raise NotImplementedError("This strategy does not support resets.")
