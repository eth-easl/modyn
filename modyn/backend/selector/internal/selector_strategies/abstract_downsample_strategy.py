# pylint: disable=singleton-comparison
# flake8: noqa: E712
import logging
import random
from typing import Any, Iterable

from modyn.backend.metadata_database.metadata_database_connection import MetadataDatabaseConnection
from modyn.backend.metadata_database.models import SelectorStateMetadata
from modyn.backend.selector.internal.selector_strategies.abstract_selection_strategy import AbstractSelectionStrategy
from sqlalchemy import asc, select

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

        assert "presampling_ratio" in config, "Please specify the ratio of presampled data"

        if self.has_limit or self.reset_after_trigger:
            raise ValueError("The current implementation only supports downsampling on the entire dataset.")

        self.presampling_ratio = config["presampling_ratio"]
        self.dataset_size = 0

        assert 0 < self.presampling_ratio <= 100

        self.ignore_presampling = self.presampling_ratio == 100

    def inform_data(self, keys: list[int], timestamps: list[int], labels: list[int]) -> None:
        assert len(keys) == len(timestamps)
        assert len(timestamps) == len(labels)

        self._persist_samples(keys, timestamps, labels)

    def _on_trigger(self) -> Iterable[list[tuple[int, float]]]:
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
        # instead of sampling B (target_size) points from the whole dataset, we sample B/num_chunks for every chunk
        self.dataset_size = self._get_dataset_size()
        assert isinstance(self.dataset_size, int)
        assert self.dataset_size > 0

        if not self.ignore_presampling:
            target_size = (self.dataset_size * self.presampling_ratio) // 100

            num_chunks = self.dataset_size // self._maximum_keys_in_memory
            num_chunks = num_chunks if self.dataset_size % self._maximum_keys_in_memory == 0 else num_chunks + 1

            per_chunk_samples = self.get_per_chunk_samples(target_size, num_chunks)
            for samples in self._get_data_no_reset_presampled(per_chunk_samples):
                random.shuffle(samples)
                yield [(sample, 1.0) for sample in samples]

        else:
            for samples in self._get_data_no_reset():
                random.shuffle(samples)
                yield [(sample, 1.0) for sample in samples]

    def get_per_chunk_samples(self, target_size: int, num_chunks: int) -> list[int]:
        last_chunk_size = self.dataset_size % self._maximum_keys_in_memory

        base_size = target_size // num_chunks
        per_chunk_samples = [base_size] * num_chunks
        per_chunk_samples[-1] = min(last_chunk_size, base_size)

        # handle remaining samples
        remaining = target_size - sum(per_chunk_samples)
        for i in range(num_chunks - 2, num_chunks - 2 - remaining, -1):
            per_chunk_samples[i] += 1

        assert sum(per_chunk_samples) == target_size

        return per_chunk_samples

    def _get_data_no_reset_presampled(self, per_chunk_samples: list[int]) -> Iterable[list[int]]:
        assert not self.reset_after_trigger
        assert sum(per_chunk_samples) > 0

        chunk_number = 0
        for samples in self._get_all_data():
            samples = random.sample(samples, min(len(samples), per_chunk_samples[chunk_number]))
            chunk_number += 1

            yield samples

    def _get_data_no_reset(self) -> Iterable[list[int]]:
        assert not self.reset_after_trigger

        for samples in self._get_all_data():
            yield samples

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
                .filter(SelectorStateMetadata.pipeline_id == self._pipeline_id)
                .order_by(asc(SelectorStateMetadata.timestamp))
            )

            for chunk in database.session.execute(stmt).partitions():
                if len(chunk) > 0:
                    yield [res[0] for res in chunk]
                else:
                    yield []

    def _reset_state(self) -> None:
        pass  # As we currently hold everything in database (#116), this currently is a noop.

    def _get_dataset_size(self) -> int:
        with MetadataDatabaseConnection(self._modyn_config) as database:
            return (
                database.session.query(SelectorStateMetadata.sample_key)
                .filter(SelectorStateMetadata.pipeline_id == self._pipeline_id)
                .count()
            )

    def get_downsampling_strategy(self) -> Any:
        """
        Abstract method to get the downsampling strategy that is transfered from the selector to the pytorch trainer
        """
        raise NotImplementedError()
