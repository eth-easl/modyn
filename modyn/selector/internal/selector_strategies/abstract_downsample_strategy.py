# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging
import random
from typing import Any, Iterable, Union

from modyn.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.metadata_database.models import SelectorStateMetadata
from modyn.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from sqlalchemy import Select, asc, func, select

logger = logging.getLogger(__name__)


class AbstractDownsampleStrategy(AbstractSelectionStrategy):
    """
    This abstract strategy is used to represent the common behaviour of downsampling strategies
    like loss-based, importance downsampling (distribution-based methods) and craig&adacore (greedy-based methods)

    These methods work on a uniformly-presampled version of the entire dataset, then the actual
    downsampling is done at the trainer server since all of these methods rely on the result of forward pass.

    Args:
        config (dict): The configuration for the selector.
    """

    def __init__(self, config: dict, modyn_config: dict, pipeline_id: int, maximum_keys_in_memory: int):
        super().__init__(config, modyn_config, pipeline_id, maximum_keys_in_memory)

        if "presampling_ratio" not in config:
            raise ValueError(
                "Please specify the presampling ratio. If you want to avoid presampling, set presampling_ratio to 100"
            )
        self.presampling_ratio = config["presampling_ratio"]

        if not (0 < self.presampling_ratio <= 100) or not isinstance(self.presampling_ratio, int):
            raise ValueError("Presampling ratio must be an integer in range (0,100]")

        if "downsampled_batch_size" not in self._config:
            raise ValueError("To use downsampling strategies, you have to specify the downsampled_batch_size")
        self.downsampled_batch_size = self._config["downsampled_batch_size"]
        if self.downsampled_batch_size <= 0 or not isinstance(self.downsampled_batch_size, int):
            raise ValueError("The downsampled batch size must be a positive integer")

        self._requires_remote_computation = True

        self.avoid_presampling = self.presampling_ratio == 100

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        self._persist_samples(keys, timestamps, labels)

    def _on_trigger(self) -> Iterable[list[tuple[int, float]]]:
        """
        Internal function. Defined by concrete strategy implementations. Calculates the next set of data to
        train on. Returns an iterator over lists, if next set of data consists of more than _maximum_keys_in_memory
        keys.

        Sampling is done within the db before chunking. Returned chunks have already been sampled

        Returns:
            Iterable[list[tuple[int, float]]]:
                Iterable over partitions. Each partition consists of a list of training samples.
                In each list, each entry is a training sample, where the first element of the tuple
                is the key, and the second element is the associated weight.
        """

        if self.avoid_presampling and not self.has_limit:
            for samples in self._get_all_data():
                random.shuffle(samples)
                yield [(sample, 1.0) for sample in samples]
        else:
            for samples in self._get_sampled_data():
                random.shuffle(samples)
                yield [(sample, 1.0) for sample in samples]

    def get_presampling_target_size(self) -> int:
        dataset_size = self._get_dataset_size()
        target_presampling = (dataset_size * self.presampling_ratio) // 100

        return target_presampling

    def _get_all_data(self) -> Iterable[list[int]]:
        """Returns all sample

        Returns:
            list[str]: Keys of used samples
        """
        with MetadataDatabaseConnection(self._modyn_config) as database:
            stmt = (
                select(SelectorStateMetadata.sample_key)
                # Enables batching of results in chunks. See https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html#orm-queryguide-yield-per
                .execution_options(yield_per=self._maximum_keys_in_memory)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id == self._next_trigger_id
                    if self.reset_after_trigger
                    else True,
                )
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def _get_sampled_data(self) -> Iterable[list[int]]:
        """Returns a subset of samples uniformly sampled from the DB

        Returns:
            list[str]: Keys of used samples
        """

        with MetadataDatabaseConnection(self._modyn_config) as database:
            if database.drivername == "postgresql":
                stmt = self.get_postgres_stmt()
            else:
                stmt = self.get_general_stmt()

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def get_postgres_stmt(self) -> Union[Select[Any], Select[tuple[Any]]]:
        # TODO(#224) write an efficient query using TABLESAMPLE
        return self.get_general_stmt()

    def get_general_stmt(self) -> Union[Select[Any], Select[tuple[Any]]]:
        presampling_target_size = self.get_presampling_target_size()

        if self.has_limit:
            target_size = min(self.training_set_size_limit, presampling_target_size)
        else:
            target_size = presampling_target_size

        subq = (
            select(SelectorStateMetadata.sample_key)
            .filter(
                SelectorStateMetadata.pipeline_id == self._pipeline_id,
                SelectorStateMetadata.seen_in_trigger_id == self._next_trigger_id if self.reset_after_trigger else True,
            )
            .order_by(func.random())  # pylint: disable=E1102
            .limit(target_size)
        )

        stmt = (
            select(SelectorStateMetadata.sample_key)
            .execution_options(yield_per=self._maximum_keys_in_memory)
            .filter(
                SelectorStateMetadata.pipeline_id == self._pipeline_id,
                SelectorStateMetadata.sample_key.in_(subq),
            )
            .order_by(asc(SelectorStateMetadata.timestamp))
        )

        return stmt

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.

    def _get_dataset_size(self) -> int:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return (
                database.session.query(SelectorStateMetadata.sample_key)
                .filter(
                    SelectorStateMetadata.pipeline_id == self._pipeline_id,
                    SelectorStateMetadata.seen_in_trigger_id == self._next_trigger_id
                    if self.reset_after_trigger
                    else True,
                )
                .count()
            )

    def get_downsampling_strategy(self) -> str:
        """
        Abstract method to get the downsampling strategy that is transfered from the selector to the pytorch trainer.
        """
        raise NotImplementedError()

    def get_downsampling_params(self) -> dict:
        """
        Abstract method to get the downsampling parameters that are transfered from the selector to the pytorch trainer.
        """
        raise NotImplementedError()
