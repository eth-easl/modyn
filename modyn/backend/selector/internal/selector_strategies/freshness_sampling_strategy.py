# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging
import random
from math import isclose

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models.metadata import Metadata
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from sqlalchemy import asc, exc, update

logger = logging.getLogger(__name__)


class FreshnessSamplingStrategy(AbstractSelectionStrategy):
    """
    This class selects data from a mixture of used and unsed data.
    We can set a ratio that defines how much data in the training set per trigger should be from previously unused data (in all previous triggers).

    The first trigger will always use only fresh data (up to the limit, if there is one).
    The subsequent triggers will sample a dataset that reflects the ratio of used/unused data (if data came during but was not used in a previous trigger, we still handle it as unseen).
    We have to respect both the ratio and the limit (if there is one) and build up the dataset on trigger accordingly.

    It cannot be used with reset, because we need to keep state over multiple triggers.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int):
        super().__init__(config, modyn_config, pipeline_id, required_configs=["unused_data_ratio"])
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

    def inform_data(self, keys: list[str], timestamps: list[int], labels: list[int]) -> None:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        # TODO(#116): Right now we persist all datapoint into DB. We might want to keep this partly in memory for performance.
        # Even if each sample is 64 byte and we see 2 million samples, it's just 128 MB of data in memory.
        with MetadataDatabaseConnection(self._modyn_config) as database:
            database.set_metadata(
                keys,
                timestamps,
                [None] * len(keys),
                [False] * len(keys),
                labels,
                [None] * len(keys),
                self._pipeline_id,
                self._next_trigger_id,
            )

    def _on_trigger(self) -> list[tuple[str, float]]:
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
            samples = self._get_first_trigger_data()
        else:
            samples = self._get_trigger_data()

        self._mark_used(samples)
        random.shuffle(samples)

        return [(sample, 1.0) for sample in samples]

    def _get_first_trigger_data(self) -> list[str]:
        assert self._is_first_trigger

        samples = self._get_all_unused_data()

        if self.has_limit and self.training_set_size_limit < len(samples):
            samples = random.sample(samples, self.training_set_size_limit)

        self._is_first_trigger = False

        return samples

    def _get_trigger_data(self) -> list[str]:
        assert not self._is_first_trigger
        unused_samples = self._get_all_unused_data()
        used_samples = self._get_all_used_data()
        # TODO(#116): At this point, we hold everything in memory anyways, so the database does not really even make sense except for very high usage scnearios

        num_unused_samples, num_used_samples = self._calc_num_samples_no_limit(len(unused_samples), len(used_samples))

        if self.has_limit and (num_unused_samples + num_used_samples) > self.training_set_size_limit:
            num_unused_samples, num_used_samples = self._calc_num_samples_limit(len(unused_samples), len(used_samples))

        return random.sample(unused_samples, num_unused_samples) + random.sample(used_samples, num_used_samples)

    def _calc_num_samples_no_limit(self, total_unused_samples: int, total_used_samples: int) -> tuple[int, int]:
        # For both the used and unsed samples, we calculate how many samples we could have at maximum where the used/unused samples make up the required fraction

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

    def _get_all_used_data(self) -> list[str]:
        """Returns all used samples

        Returns:
            list[str]: Keys of used samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(Metadata.pipeline_id == self._pipeline_id, Metadata.seen == True)
                .all()
            )

        if len(data) > 0:
            keys, seen = zip(*data)
            assert all(seen), "Queried seen data, but got unseen data."
        else:
            keys, seen = [], []

        return list(keys)

    def _get_all_unused_data(self) -> list[str]:
        """Returns all unused samples

        Returns:
            list[str]: Keys of unused samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            data = (
                database.session.query(Metadata.key, Metadata.seen)
                .filter(Metadata.pipeline_id == self._pipeline_id, Metadata.seen == False)
                .order_by(asc(Metadata.timestamp))
                .all()
            )

        if len(data) > 0:
            keys, seen = zip(*data)
            assert not any(seen), "Queried unseen data, but got seen data."
        else:
            keys, seen = [], []

        return list(keys)

    def _mark_used(self, keys: list[str]) -> None:
        """Sets samples to used"""
        if len(keys) == 0:
            return

        with MetadataDatabaseConnection(self._modyn_config) as database:
            try:
                stmt = update(Metadata).where(Metadata.key.in_(keys)).values(seen=True)
                database.session.execute(stmt)
                database.session.commit()
            except exc.SQLAlchemyError as exception:
                logger.error(f"Could not set metadata: {exception}")
                database.session.rollback()

    def _reset_state(self) -> None:
        raise NotImplementedError("This strategy does not support resets.")
